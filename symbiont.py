from __future__ import annotations

import logging
import copy  # for copy constructor:
    # http://stackoverflow.com/questions/1241148/copy-constructor-in-python
    # http://pymotw.com/2/copy/
import sys   # for sys.exit
from numpy import cumsum
from parameters import *
from rng_mt19937 import *
from event_list import Event, EventList
from clade import *
from sponge import *
from enums import EventType, SymbiontState

###############################################################################
# This class implements a symbiont alga in the agent-based simulation.
# Each symbiont will have the following instance variables, as well as those
# contained by its parent class Clade:
# 
#    self._id                      : integer count of this symbiont
#    self._cell                    : cell of residence (none if evicted)
#    self._mitotic_cost_rate       : cost of mitosis (photosynthate per unit time)
#    self._production_rate         : photosynthetic production rate (per unit time)
#    self._photosynthate_surplus   : banked photosynthate (e.g., to use for mitosis)
#    self._surplus_on_arrival      : photosynthate on arrival (from pool or inherited)
#
#    self._arrival_time            : when this symbiont appeared in the system
#    self._prev_event_time         : last time this symbiont did something
#    self._prev_event_type         : last type of thing this symbiont did 
#    self._time_of_escape          : time when symbiont is exiting quickly
#    self._time_of_digestion       : time when symbiont will be digested
#    self._time_of_denouement      : time when sybmiont leaves on own accord
#    self._time_of_next_end_g0     : time when next G0 period ends
#    self._time_of_next_end_g1sg2m : time when next mitosis ends
#
#    self._next_event_time         : next (min) event time among those above
#    self._next_event_type         : event type associated with the next event
#    self._prev_event_time         : previous (min) event time among those above
#    self._prev_event_type         : event type associated with the previous event
######################
#    bgl: Feb|Mar 2024
#    self._prev_update_time          : previous time photosynthate was updated
#                                      (NB: with multiple symbionts per cell, each 
#                                      must have photosynthate surplus updated at the
#                                      end of any event for any symbiont in this cell)
######################
#
#    self._how_arrived             : ARRIVED_FROM_POOL (0), ARRIVED_VIA_DIVISION (1)
#    self._parent_id               : id of parent, -1 if arrived from pool
#    self._agent_zero              : ultimate ancestor (may be self)
#    self._surplus_on_arrival      : photosynthate on hand when arrived
#    self._num_divisions           : number of successful divisions for symbiont
#    self._clade_number            : the number ID of the clade of this symbiont
#    self._my_clade                : an instance of the parent Clade of this symbiont
#
#    self._cells_at_division       : list of number of open cells available at each mitosis
#    self._cells_inhabited         : list of all cells inhabited
#    self._inhabit_times           : list of times each cell above inhabited
#    self._hcds_of_cells_inhabited : list of photosynthate demand of each cell inhabited
#    self._g0_times                : list of g0 times for this symbiont
#    self._g1sg2m_times            : list of g1sg2m times for this symbiont
#    self._num_divisions           : number of successful divisions for this symbiont
#
# Methods of interest (all but one public-facing) used in the simulation:
#    __init__                          : construct a new arrival from pool
#    _SymbiontCopy(cell, current_time) : construct a new arrival from mitotsis
#    endOfG0(t)                        : handle end of G0 period
#    endOfG1SG2M(t)                    : handle end of G1SG2M, when division occurs
#    digestion(t)                      : use when symbiont digestion occurs
#    escape(t)                         : use when symbiont escape occurs
#    denouement(t)                     : use when symbiont denouement occurs
#    __str__()                         : use to print a symbiont
#    getNextSymbiontEvent()            : returns next event as (time,EventType)
#    openCSVFile        [class-level]  : open CSV file for writing per-symbiont info (if requested)
#    csvOutputOnExit    [class-level]  : dumps symbiont info to CSV @ time of symbiont exit
#    csvOutputAtEnd     [class-level]  : dumps remaining in-residence symbiont info to CSV @ simulation end
#    findOpenCell       [class-level]  : finds an open cell at random among all avaiable in sponge
#    findOpenCellWithin [class-level]  : finds an open cell at random within a given neighborhood
#    generateArrival    [class-level]  : generates a symbiont arrival, if space in the sponge
#
# Notes on Affinity Terms (Spring 2019):
#   ARRIVAL AFFINITY - the level of coevolution between symbiont and host in 
#   regards to the ability of the symbiont to safely enter the host cell. 
#   Can take the form of a number of different strategies based on different
#   biological features that compose it.
# ----------------------------------------------------------------------------
#   DIVISION AFFINITY - the level of coevolution between symbiont and host in 
#   regards to the ability of the symbiont to safely maintain presence in the
#   host cell. Can take the form of a number of different strategies based on 
#   different biological features that compose it.
###############################################################################

################################################################################
class Symbiont:
    # the __slots__ tuple defines the names of the instance variables for a 
    # Symbiont object so that, e.g.,  mistyping a name doesn't accidentally
    # introduce a new instance variable (a Python "feature" if using the 
    # default dict approach without __slots__ defined)
    __slots__ = ( \
                 '_agent_zero',              \
                 '_arrival_time',            \
                 '_cell',                    \
                 '_cells_at_division',       \
                 '_cells_inhabited',         \
                 '_clade_number',            \
                 '_g0_times',                \
                 '_g1sg2m_times',            \
                 '_hcds_of_cells_inhabited', \
                 '_how_arrived',             \
                 '_id',                      \
                 '_inhabit_times',           \
                 '_mitotic_cost_rate',       \
                 '_my_clade',                \
                 '_next_event_time',         \
                 '_next_event_type',         \
                 '_num_divisions',           \
                 '_parent_id',               \
                 '_photosynthate_surplus',   \
                 '_prev_event_time',         \
                 '_prev_event_type',         \
                 #############################
                 # bgl: Feb|Mar 2024
                 '_prev_photosynthate_change', \
                 '_prev_update_time',        \
                 #############################
                 '_production_rate',         \
                 '_surplus_on_arrival',      \
                 '_time_of_escape',          \
                 '_time_of_digestion',       \
                 '_time_of_denouement',      \
                 '_time_of_next_end_g0',     \
                 '_time_of_next_end_g1sg2m', \
                 )

    # class-level variables
    sponge : Sponge = None   # set in simulation.py main function when symbionts are created
    clade_cumulative_proportions : list[float] = [None] * len(Parameters.CLADE_PROPORTIONS)
    _count : int      = 0      # used to count total number of symbionts

    # class-level variables for writing per-symbiont statistics; whether to
    # write the CSV information can be changed via command-line argument at
    # simulation execution
    _write_csv:  bool                = False  
    _csv_writes: int                 = 0
    _csv_file:   '_io.TextIOWrapper' = None

    ############################################################################
    def __init__(self, clade_number: int, cell: Cell, current_time: float) -> None:
        ''' initializer used to create symbionts that arrive from the pool
            (oustide)
        Parameters:
            clade_number: integer corresponding to the specific clade of this symbiont
            cell: Cell object into which this symbiont will be arriving
            current_time: floating point value of current simulation time
        Raises:
            RuntimeError, if the Symbiont class-level sponge variable has not been set
            ValueError, if the provided clade number is invalid
        '''
        # Error checking
        if Symbiont.sponge is None:
            raise RuntimeError(f"Error in Symbiont: class-level sponge environment not set")
        if clade_number < 0 or clade_number >= Parameters.NUM_CLADES:
            raise ValueError(f"Error in Symbiont: invalid clade {clade_number}")

        ########################################################################
        # define instance variables and type (hints) so they are in one place
        # for easy reference; actual values will be assigned below and/or later
        self._id:                      int           = None
        self._clade_number:            int           = None
        self._my_clade:                Clade         = None
        self._cell:                    Cell          = None
        self._how_arrived:             SymbiontState = None
        self._parent_id:               int           = None
        self._agent_zero:              int           = None
        self._num_divisions:           int           = None
        self._mitotic_cost_rate:       float         = None
        self._production_rate:         float         = None
        self._photosynthate_surplus:   float         = None
        self._surplus_on_arrival:      float         = None
        self._cells_inhabited:         list[str]     = None
        self._inhabit_times:           list[float]   = None
        self._hcds_of_cells_inhabited: list[float]   = None 
        self._g0_times:                list[float]   = None
        self._g1sg2m_times:            list[float]   = None
        #self._cells_at_division:       list[int]     = None

        self._arrival_time:            float         = None
        self._time_of_escape:          float         = None
        self._time_of_digestion:       float         = None
        self._time_of_denouement:      float         = None
        self._time_of_next_end_g0:     float         = None
        self._time_of_next_end_g1sg2m: float         = None

        self._prev_event_time:         float         = None
        self._prev_event_type:         EventType     = None
        self._next_event_time:         float         = None
        self._next_event_type:         EventType     = None
        # bgl: Feb|Mar 2024
        self._prev_photosynthate_change: float       = None
        self._prev_update_time:          float       = None
        ########################################################################

        self._id = Symbiont._count
        Symbiont._count += 1

        self._clade_number  = clade_number
        self._my_clade      = Clade.getClade(clade_number)
        self._cell          = cell
        self._how_arrived   = SymbiontState.ARRIVED_FROM_POOL
        self._parent_id     = -1        # arriving from pool, no parent
        self._agent_zero    = self._id  # arriving from pool, topmost pregenitor is self
        self._num_divisions = 0

        # INDIVIDUAL SYMBIONT FUZZING
        # Rather than fuzzing uniformly, use normal with 95% of the data
        # between (mu +/- mu*f) -- see implementation in rng.py
        m = float(self._my_clade.getMCR())
        f = float(self._my_clade.getMCRFuzz())  # assume to be % of the mean
        self._mitotic_cost_rate = RNG.fuzz(m, f, Stream.MITOTIC_COST_RATE)

        self._production_rate = self._computeProductionRate(is_copy = False)
            #self._computeProductionRate(is_copy = False, current_time = current_time)

        # 21 Apr 2016: change initial photosynthate to use a Gamma distribution;
        # Connor and Barry experimented and determined gamma(2,0.75) looks
        # reasonable -- mean = 1.5, 50% = 1.25
        # 08 Oct 2016: add max, per Malcolm suggetion in 7 Oct meeting
        clade_max_photosynthate = self._my_clade.getMaxInitialSurplus()
        self._photosynthate_surplus = INFINITY
        while self._photosynthate_surplus > clade_max_photosynthate:
            self._photosynthate_surplus = RNG.gamma( \
                self._my_clade.getInitialSurplusShape(), \
                self._my_clade.getInitialSurplusScale(), \
                Stream.PHOTOSYNTHATE)
        #print(">>>>>>>>> ORIG SURPLUS= ",self._photosynthate_surplus)

        self._surplus_on_arrival = self._photosynthate_surplus

        # 22 Sep 2016: Malcolm wanted to keep track of total residence time per
        # symbiont as well as residence time per cell for a symbiont (e.g., when
        # parent divides and parent moves to a different cell) -- the following
        # data structure is to keep track of times when a symbiont does not exit
        # the system but moves into a different cell within our grid
        #
        # 5 Oct 2016: also want a list of all cells visited so we can dump that
        # info into CSV at end, but more importantly the hcd for those cells
        # (looking for evolutionary advantages given symbionts)...
        self._cells_inhabited         = [str(cell.getLevelRowCol()).replace(', ',',')]
        self._inhabit_times           = [current_time]
        self._hcds_of_cells_inhabited = [cell.getDemand()]

        # 5 Oct 2016: Malcolm also wanted to keep track of the g0 and g1sg2m
        # lengths for symbionts, to look for evolutionary advantage; these
        # times are created at random with each g0 and g1sg2m event, so we need
        # to also store all of them in a list per symbiont
        self._g0_times = []
        self._g1sg2m_times = []

        # 13 Feb 2017: also track the # of open cells around at time of division
        #self._cells_at_division = []

        self._arrival_time              = current_time  # when arrived -- now!
        self._prev_event_time           = current_time
        self._prev_event_type           = EventType.ARRIVAL

        # bgl: Feb|Mar 2024
        self._prev_photosynthate_change = 0.0           # no prev change/adjustment to start
        self._prev_update_time          = current_time  # first prev photoshyntate update was now

        self._time_of_escape            = INFINITY      # global in Parameters
        self._time_of_digestion         = INFINITY  
        self._time_of_denouement        = INFINITY
        self._time_of_next_end_g0       = INFINITY      # end of G0, start of G1/S/G2/M
        self._time_of_next_end_g1sg2m   = INFINITY      # end of G1/S/G2/M, start of G0

        self._next_event_time           = INFINITY      # set in scheduling events next...
        self._next_event_type           = None

        self._scheduleInitialEvents(current_time)
        self._setNextEvent()

    #############################################################################
    def _SymbiontCopy(self, cell: Cell or None, current_time: float) -> Symbiont:
        ''' This "copy constructor" is used to create symbionts that occur from
            mitosis.  The resulting symbiont should have the same clade.  The
            copy will have its own id, own event times, and (depending on
            row-location of the host cell) potentially different photosynthate
            production rate.
        Parameters:
            cell: the Cell object into which the new symbiont will go, or None
                if no cell available or if infecting outside the model's scope
            current_time: floating point value of current simulation time

        Returns:
            a newly copied and modified Symbiont object resulting from mitosis
        '''
        new_symbiont = copy.copy(self)  # make an exact copy of this symbiont

        # now begin updating its values as a symbiont arriving anew from mitosis
        new_symbiont._id = Symbiont._count
        Symbiont._count += 1

        new_symbiont._num_divisions = 0

        new_symbiont._how_arrived     = SymbiontState.ARRIVED_VIA_DIVISION
        new_symbiont._parent_id       = self._id          # new symbiont's parent is this symbiont
        new_symbiont._agent_zero      = self._agent_zero  # same original progenitor as self

        new_symbiont._cell            = cell
        new_symbiont._arrival_time    = current_time  # when arrived -- now!
        new_symbiont._prev_event_time = current_time
        new_symbiont._prev_event_type = EventType.ARRIVAL

        # 23 Sep 2016 and 5 Oct 2016 and 13 Feb 2017:
        # clear out any switched times inherited from parent
        new_symbiont._cells_inhabited         = []  # may be updated below...
        new_symbiont._inhabit_times           = []
        new_symbiont._hcds_of_cells_inhabited = []
        new_symbiont._g0_times                = []
        new_symbiont._g1sg2m_times            = []
        #new_symbiont._cells_at_division       = []

        ## 12 Apr 2016
        # fuzz all of the inherited values (MCR, PPR, surplus)
        # use normal with 95% of the data b/w (mu +/- mu*f) -- see rng.py;
        # MCR is fuzzed here; surplus below; PPR inside _computeProductionRate()
        #####################################################
        assert(new_symbiont._mitotic_cost_rate == self._mitotic_cost_rate)
        m = new_symbiont._mitotic_cost_rate

        #####################################################
        ## OLD WAY -- using normal
        '''
        f = Parameters.DIV_FUZZ  # assume to be % of the mean
        new_symbiont._mitotic_cost_rate = RNG.fuzz(m, f, Stream.MITOTIC_COST_RATE)
        '''
        ## NEW WAY -- using combined gamma for deleterious & beneificial
        [fuzzamt, mutation] = RNG.divfuzz(m, self._my_clade, Stream.MITOTIC_COST_RATE_MUTATION)
        if mutation == MutationType.DELETERIOUS:
            new_symbiont._mitotic_cost_rate += fuzzamt  # deleterious mcr increases
        else:  # mutation == MutationType.BENEFICIAL:
            new_symbiont._mitotic_cost_rate -= fuzzamt  # beneficial mcr decreases

        # uncomment below if want to see info about mutations...
        '''
        if mutation != MutationType.NO_MUTATION:
            mut_type = 'DEL' if mutation == MutationType.DELETERIOUS else 'BEN'
            print(f'@ t={current_time} {new_symbiont._id} {mut_type} mcr: {m} {new_symbiont._mitotic_cost_rate}')
        '''

        # give up roughly half (fuzzed) of the banked photosynthate to new symbiont
        half = self._photosynthate_surplus / 2
        #old = self._photosynthate_surplus  # for testing whether halving works correctly (below)

        ## OLD WAY -- using normal
        '''
        m = half
        f = Parameters.DIV_FUZZ
        half = RNG.fuzz(m, f, Stream.PHOTOSYNTHATE)
        fuzzedhalf = RNG.divfuzz(half, Stream.PHOTOSYNTHATE)
        '''
        ## NEW WAY -- using combined exponential for deleterious & beneficial
        [fuzzamt, mutation] = RNG.divfuzz(half, self._my_clade, Stream.PHOTOSYNTHATE_MUTATION)
        fuzzedhalf = half
        if mutation == MutationType.DELETERIOUS:
            fuzzedhalf -= fuzzamt  # deleterious inheritance slightly less than half
        elif mutation == MutationType.BENEFICIAL:
            fuzzedhalf += fuzzamt  # beneficial inheritance slightly more than half

        # uncomment below if want to see info about mutations...
        '''
        if mutation != MutationType.NO_MUTATION:
            mut_type = 'DEL' if mutation == MutationType.DELETERIOUS else 'BEN'
            print(f'@ t={current_time} {new_symbiont._id} {mut_type} surplus: {half} {fuzzedhalf}')
        '''

        #print(f">>>>>>>>> FUZZED INHERIT SURPLUS = {new_symbiont._photosynthate_surplus}")

        new_symbiont._photosynthate_surplus = fuzzedhalf     # child gets (fuzzed) half
        self._photosynthate_surplus -= fuzzedhalf            # parent decremented by half

        new_symbiont._surplus_on_arrival = new_symbiont._photosynthate_surplus

        #print(">>>>>>>>> HALVING WORKS?= ", \
        #    (old == new_symbiont._photosynthate_surplus + self._photosynthate_surplus))

        # this divided cell may not have a cell to reside in or may be infecting
        # outside our environment, and if so, there is no need to compute a
        # production rate or initial events...
        # (calling the methods below will actually cause a problem)
        if new_symbiont._cell is not None:
            assert(new_symbiont._production_rate == self._production_rate)

            new_symbiont._production_rate = \
                new_symbiont._computeProductionRate(is_copy = True)  # fuzzing inside!
                #new_symbiont._computeProductionRate(is_copy = True, current_time = current_time)  # fuzzing inside!

            new_symbiont._scheduleInitialEvents(current_time)
            new_symbiont._setNextEvent()

            # 5 Oct 2016
            new_symbiont._cells_inhabited         = [str(new_symbiont._cell.getLevelRowCol()).replace(', ',',')]
            new_symbiont._inhabit_times           = [current_time]
            new_symbiont._hcds_of_cells_inhabited = [new_symbiont._cell.getDemand()]

        return new_symbiont # return the newly created copy

    ##############################################################################################
    # bgl: Feb|Mar 2024
    def _rollBack(self) -> None:
        ''' private method to roll back this symbiont's ongoing photosynthate production/
            consumption to its previous change point, by simply adding back the amount by
            which the photosynthate bank was changed at the previous change time
        '''    
        self._photosynthate_surplus += self._prev_photosynthate_change

    def _getState(self) -> SymbiontState:
        ''' determine the current state of the symbiont based on its previous
            event type
        Returns:
            the current SymbiontState (see enums.py)
        '''
        if self._prev_event_type == EventType.ARRIVAL:    return SymbiontState.IN_G0
        if self._prev_event_type == EventType.END_G0:     return SymbiontState.IN_G1SG2M
        if self._prev_event_type == EventType.END_G1SG2M: return SymbiontState.IN_G0
        # should never get here -- an escaping or digested or departing symbiont should
        # not be calling this method...
        assert(False)

    ##############################################################################################
    # bgl: Feb|Mar 2024
    def cancelEventWithinSymbiont(self, event_type: EventType) -> None:
        ''' method to cancel a future digestion or escape event, updating the
            next event to instead be whatever event would have happened had 
            the digestion|escape not been scheduled (i.e., end G0, end G1SG2M,
            denouement)
        Parameters:
            event_type: type of event to be cancelled (now only digestion or escape)
        '''
        assert(event_type in [EventType.DIGESTION, EventType.ESCAPE])
        # THIS WILL HAVE TO CHANGE -- ON DIGESTION OR ESCAPE, MAY HAVE TO CANCEL
        # THE G0 AND G1 as the remaining symbionts now may not be able to make
        # it past 
        # NB: make sure to remove the event from the event list
        #       but not in the symbiont (maybe reuse that end G0, e.g.)

        if event_type == EventType.DIGESTION: self._time_of_digestion = INFINITY
        if event_type == EventType.ESCAPE:    self._time_of_escape    = INFINITY
        # now update this symbiont's next event to whatever would have happened
        # in lieu of the digestion or escape
        self._setNextEvent()


    ##############################################################################################
    # bgl: Feb|Mar 2024
    #def _computeSurplusAtEventEnd(self, this_time: float, next_time: float, state: SymbiontState) \
    # rla: Feb|Mar 2024 - changed method to accept list of symbionts as a parameter,
    #                     rather than number of symbionts
    def _computeSurplusAtEventEnd(self, this_time: float, 
                                        next_time: float, 
                                        symbionts_in_cell: list) \
            -> list[float, float or None, float or None]:
        ''' This method computes the amount of photosynthate surplus that will
            be present at the end of the next event, indicating whether the
            symbiont will make it successfully to the event or whether we
            should schedule an exit strategy for the symbiont.  
        Parameters:
            this_time: the current event time (float)
            next_time: the next minimum event time for the cell occupied by
                this symbiont [which may correspond to an event time for a
                different symbiont also in the same cell] (float)
            ### removed by rla - 3/22 ###
            num_symbionts_in_cell: the current number of symbionts present in
                the cell, for computing shared host cell demand load
                    (NB: on arrival, the arriving symbiont has not yet been
                    officially added to the cell [see end of @classmethod
                    generateArrival], which should have been accounted for
                    in the argument when calling this method)
            #############################
            symbionts_in_cell: the current list of symbionts present in the cell,
                for computing shared host cell demand load
                (note: with the current equal-share-load implementation, we only 
                need the number of symbionts in a given cell to compute the surplus;
                future implementations may require access to each symbiont)
        Returns:
            a list containing:
                (a) the computed surplus at the end of the event
                (b) the computed digestion time (or None, if not being digested)
                (c) an exit expulsion time (or None, if not exiting)
        '''

        # if this symbiont can't produce photosynthate at a rate sufficient
        # to meet the host cell's demand through the next event, will need
        # to eventually (elsewhere) schedule the exit strategy...
        time_diff = next_time - this_time
        produced = time_diff * self._production_rate
        # rla: Feb|Mar 2024 -- for equal-share-load implementation
        num_symbionts_in_cell = len(symbionts_in_cell)
        # bgl: Feb|Mar 2024
        #demanded = time_diff * self._cell.getDemand()
        demanded = time_diff * (self._cell.getDemand() / num_symbionts_in_cell)
        expended = 0

        # only during mitosis (coming out of G1SG2M) should we compute and
        # expended photosynthate as a cost for undergoing mitosis (state will
        # be IN_G1SG2M)

        # bgl: Feb|Mar 2024
        # NB: read the corresponding comment in endOfG0 about ensuring that
        #     the call to _getState() here will return the right thing when
        #     calling _projectToSymbiontNextEvent at the last of endOfG0
        #if state == SymbiontState.IN_G1SG2M:
        if self._getState() == SymbiontState.IN_G1SG2M:
            expended = time_diff * self._mitotic_cost_rate

        surplus_at_end = self._photosynthate_surplus + produced - demanded - expended

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        hcd = self._cell.getDemand()
        #print(f"{self._id}: t = {this_time:.4f}  next_t = {next_time:.4f}  " + \
        #      f"symb next_t = {self._next_event_time:.4f}  "
        #      f"dmnd rt = {hcd:.4f}  prod = {produced:.4f}  dmnd = {demanded:.4f}  surp = {surplus_at_end:.4f}")
        #print(f"{self._id}: t={this_time:>6.3f} cell,sym={next_time:>6.3f},{self._next_event_time:^6.3f} "
        #      f"hcd={hcd:>6.3f} prod={produced:>6.3f} dmd={demanded:>6.3f} srp={surplus_at_end:>6.3f}")
        print(f"{self._id}: t={this_time:>6.4f} cell_t,sym_t={next_time:>6.4f},{self._next_event_time:^6.4f}\n" +
              f"\thcd={hcd:>6.4f} prod={produced:>6.4f} dmd={demanded:>6.4f} mit={expended:>6.4f} srp={surplus_at_end:>6.4f}")
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        '''
        logging.debug(f'\t>>>time of next event: {next_time}')
        logging.debug(f'\t>>>produced:           {produced}')
        logging.debug(f'\t>>>demanded:           {demanded}')
        logging.debug(f'\t>>>expended:           {expended}')
        logging.debug(f'\t>>>surplus then:       {surplus_at_end}')
        '''

        t_d  = None # time of digestion, if any computed below
        t_ee = None # time of exit expulsion, if any computed below

        if surplus_at_end < 0:
            # the symbiont will not survive through the next event in this cell;
            # so, determine when the surplus will drop below 0 and then
            # schedule a digestion event or, if lucky with the coin flip, an
            # exit-expulsion; consider endpoints of line (t_c,s_c) and (t_e,s_e)
            # where 'c' is current, 'e' is end, and s_e < 0;  then slope of line
            # is
            #         m = (s_e - s_c) / (t_e - t_c)
            # and the computed time of digestion t_d can be computed by solving
            # y - y1 = m(x - x1) using (t_c,s_c) and solving for x when y = 0:
            #         t_d = t_c - (s_c/m)
            t_c = this_time
            s_c = self._photosynthate_surplus
            t_e = next_time
            s_e = surplus_at_end
            assert((t_e - t_c) > 0.0)
            m   = (s_e - s_c) / float(t_e - t_c)
            t_d = t_c - (s_c / m)  # computed time of digestion
            #logging.debug(f'\t>>>time of digestion:  {t_d}')

            # determine if this to-be-digested symbiont is lucky enough to
            # instead exit first; use the correct stream and probability...
            # bgl: Feb|Mar 2024
            #if state == SymbiontState.IN_G0:
            if self._getState() == SymbiontState.IN_G0:
                stream_prob = Stream.DIGESTION_VS_ESCAPE_G0
                prob        = self._my_clade.getG0EscapeProb()
                stream_exit = Stream.TIME_G0_ESCAPE
            # bgl: Feb|Mar 2024
            #elif state == SymbiontState.IN_G1SG2M:
            elif self._getstate() == SymbiontState.IN_G1SG2M:
                stream_prob = Stream.DIGESTION_VS_ESCAPE_G1SG2M
                prob        = self._my_clade.getG1SG2MEscapeProb()
                stream_exit = Stream.TIME_G1SG2M_ESCAPE
            else:
                assert(False) # should never get here if state is not one of the above

            p = RNG.uniform(0, 1, stream_prob)
            if p < prob:
                # lucky -- will have an exit expulsion before digesting
                t_ee = RNG.uniform(this_time, t_d, stream_exit)

        if t_d is not None or t_ee is not None:
            assert(surplus_at_end < 0) # sanity check

        return [surplus_at_end, t_d, t_ee]

    #############################################################################
    def _computeNextEndOfG0(self, current_time: float) -> float:
        ''' Method to compute the next time of an end-of-G0 event, using
                current_time + avg G0 length +/- small fudge
        Parameters:
            current_time: the current event (simulation) time (float)
        Returns:
            the next end-of-G0 time for this symbiont (float)
        '''
        # using normal distribution
        m = self._my_clade.getG0Length()
        f = self._my_clade.getG0Fuzz()
        g0_time = RNG.fuzz(m, f, Stream.END_G0)  # fuzzed version of G0 length
        #print(f">>> G0: {g0_time}")

        next_time = current_time + g0_time
        self._g0_times.append(g0_time)
        return next_time

    #############################################################################
    def _computeNextEndOfG1SG2M(self, current_time: float) -> float:
        ''' Method to compute the next time of an end-of-G1SG2m event, using
                current_time + avg G1SG2M length +/- small fudge
        Parameters:
            current_time: the current event (simulation) time (float)
        Returns:
            the next end-of-G1SG2M time for this symbiont (float)
        '''
        # Using normal distribution
        m = self._my_clade.getG1SG2MLength()
        f = self._my_clade.getG1SG2MFuzz()
        g1sg2m_time = RNG.fuzz(m, f, Stream.END_G1SG2M)  # fuzzed version of G1SG2M length
        #print(f">>> G1SG2M: {g1sg2m_time}")

        next_time = current_time + g1sg2m_time
        self._g1sg2m_times.append(g1sg2m_time)
        return next_time

    #############################################################################
    # bgl: Feb|Mar 2024
    def _projectToSymbiontNextEvent(self, 
                                    current_time: float, 
                                    next_event_time: float, 
                                    next_event_type: EventType) -> None:
        ''' private method for simply projecting the photosynthate surplus at
            the end of the next _actual_ event time for this symbiont (not just
            a photosynthate update, i.e., event time for some other symbiont in
            this same cell);  if the symbiont can successfully make the end of
            the event, that event type is scheduled for the symbiont;  if the
            symbiont cannot successfully make the end of the event, either a
            digestion or escape is scheduled for the symbiont;
            NB: no actual updated of the phoysynthate surplus happens here --
                that occurs in _projectSurplusForAll

        Parameters:
            current_time:    time of the current event (either end G0 or G1SG2M)
            next_event_time: time of the next event (oppo of end G0 vs G1SG2M)
            next_event_type: type of event corresponding to next_event_time
        '''

        # sanity check
        assert(next_event_type in [EventType.END_G0, EventType.END_G1SG2M])

        # rla: Feb|Mar 2024
        # num_symbionts_in_cell = self._cell.getNumOccupants()
        symbionts_in_cell = self._cell.getSymbionts()

        # now, compute the amount of photosynthate produced, demanded by the
        # host cell (shared across all symbionts in this cell), and, if in
        # G1SG2M, expended on mitosis during the entire period for this symbiont
        # [surplus_at_end, time_of_digestion, time_of_exit] = \
        #    self._computeSurplusAtEventEnd(current_time, next_event_time, \
        #                num_symbionts_in_cell)
        [surplus_at_end, time_of_digestion, time_of_exit] = \
            self._computeSurplusAtEventEnd(current_time, next_event_time, \
                        symbionts_in_cell)

        if surplus_at_end < 0:
            # the event will not complete successfully in this cell environment; 
            # so, determine when the surplus will drop below 0 and then schedule a 
            # digestion event or, if lucky with the coin flip, an exit-expulsion as 
            # the mitosis event is progressing...
            self._time_of_digestion = time_of_digestion
            if time_of_exit is not None:
                assert(time_of_digestion == INFINITY)
                self._time_of_escape = time_of_exit
        else:
            # the event will complete successfully, so schedule it.
            # NB: the _next_ event type (not current) is end of G0
            if   next_event_type == EventType.END_G0:
                self._time_of_next_end_g0     = next_event_time
            elif next_event_type == EventType.END_G1SG2M:
                self._time_of_next_end_g1sg2m = next_event_time
            else:
                assert(False)  # sanity check
            # WILL NEED TO ADD DENOUEMENT


    #############################################################################
    # bgl: Feb|Mar 2024
    def _projectSurplusForAll(self, 
                next_event_time:   float,
                update_events:     bool     = False,  
                update_bank:       bool     = False,
                arriving_symbiont: Symbiont = None) -> None:
        ''' private method for computing/updating --- for all symbionts in the
            current cell --- the photosynthate projections (i.e., the amount of
            photosynthate left after the symbiont produces as well as provides
            for [shared] host cell demand) across a given time period;  an
            actual update to each symbiont's photosynthate bank happens at the
            start of any event for any symbiont in the cell, and will be noted
            by the corresponding parameter
        Parameters:
            next_event_time: the time of the next event occurring in this cell
            update_events: if True, update events within each symbiont should
                the projection determine that a digestion or exit should be
                scheuled
            update_bank: if True, actually enact the update to each symbiont's
                photosynthate bank; o/w, this is used as a projection to ensure
                the symbiont makes it through to the next event occuring in
                the current cell
            arriving_sybmiont: if not None, corresponds to the now-arriving
                symbiont which has not yet been added to the cell's list of
                symbionts
        '''
        # when processing _any_ event, need to update the photosynthetic
        # production projection for _all_ symbionts in that cell
        cell_symbionts = self._cell.getSymbionts()  # returns a copy
        if arriving_symbiont is not None:
            assert(arriving_symbiont not in cell_symbionts)
            assert(arriving_symbiont == self)
            cell_symbionts.append(arriving_symbiont)
        # rla: Feb|Mar 2024 -- use cell_symbionts instead
        # num_symbionts_in_cell = len(cell_symbionts)

        for symb in cell_symbionts:
            #[surplus_at_end, time_of_digestion, time_of_exit] = \
            #    symb._computeSurplusAtEventEnd(symb._prev_update_time, next_event_time, \
            #            num_symbionts_in_cell)
            [surplus_at_end, time_of_digestion, time_of_exit] = \
                symb._computeSurplusAtEventEnd(symb._prev_update_time, next_event_time, \
                        cell_symbionts)

            # sanity check
            if not update_events and surplus_at_end < 0:
                print("Not updating events in _projectSurplusForAll yet " + \
                      "digestion or escape is warranted")
                assert(False)

            if update_events and surplus_at_end < 0:
                # the symbiont will not survive through the next event for this
                # cell, so schedule a digestion event or, if lucky with a
                # coin flip, an escape
                symb._time_of_digestion = time_of_digestion
                if time_of_exit is not None:
                    # rapid expulsion, uniformly distributed between now 
                    # and when the digestion would naturally occur
                    symb._time_of_escape = time_of_exit

            # this method may be called either to project ahead, or to do the 
            # actual bank update -- update the surplus only when appropriate
            if update_bank:
                symb._photosynthate_surplus = surplus_at_end
                symb._prev_update_time = next_event_time

    #############################################################################
    def endOfG0(self, current_time: float) -> None:
        ''' Method to handle the transition from end of G0 into G1SG2M.  This
            method will compute the amount of photosynthate surplus the symbiont
            will have before the end of the coming G1SG2M event. If all goes 
            well for the symbiont, the surplus will be positive and an end-of-
            G1SG2M event can be scheduled.  If the surplus is determined to be
            negative by the time of the future end-of-G1SG2M event, the symbiont
            will either been digested or will have escaped prior -- so need to
            schedule that earlier event instead.
        Parameters:
            current_time: the current simulation time -- @ end of G0 (float)
        '''

        self._time_of_next_end_g0 = INFINITY
        assert(self._prev_event_type == EventType.ARRIVAL or \
               self._prev_event_type == EventType.END_G1SG2M)

        # bgl: Feb|Mar 2024
        assert(self._getState() == SymbiontState.IN_G0)

        # bgl: Feb|Mar 2024
        # when processing _any_ event, need to update the photosynthetic
        # production projection for _all_ symbionts in that cell (each had
        # already projected to ensure getting at least to this event, so update
        # the bank of each symbiont in the current cell)
        self._projectSurplusForAll(current_time, update_bank = True)

        # now, compute the amount of photosynthate produced, demanded by the
        # host cell, and expended on mitosis during the entire G1/S/G2/M period
        # for _this_ symbiont
        time_of_end_g1sg2m = self._computeNextEndOfG1SG2M(current_time)

        # bgl: Feb|Mar 2024
        # _projectToSymbiontNextEvent will project out across the event time
        # and if the symbiont cannot successfully complete, will schedule a
        # digestion or escape event; if can complete, will schedule end of
        # G1SG2M for this symbiont
        print('##################### BEFORE')
        # NB: _prev_event_{time,type} need to be set before calling 
        #     _projectToSymbiontNextEvent so that method will account for
        #     the transition into mitosis (i.e., will deduct additionally,
        #     by way of its call to _computeSurplusAtEventEnd)
        #     --> in other words, setting these here causes _getState() to 
        #         return SymbiontState.IN_G1SG2M appropriately
        self._prev_event_time = current_time
        self._prev_event_type = EventType.END_G0

        self._projectToSymbiontNextEvent(
                             current_time    = current_time,
                             next_event_time = time_of_end_g1sg2m,
                             next_event_type = EventType.END_G1SG2M)
        print('##################### AFTER')

        # whatever the event might be, set the next event for this symbiont
        # bgl: Feb|Mar 2024
        #self._prev_event_time = current_time
        #self._prev_event_type = EventType.END_G0
        self._setNextEvent()
        # bgl: Feb|Mar 2024
        self._cell.updateNextEventTime()

    #############################################################################
    def endOfG1SG2M(self, current_time: float) -> list[SymbiontState, Symbiont]:
        ''' Method to handle the transition from end of G1SG2M, when the mitosis
            is completing, into the next G0 state.  This method checks for all
            possible scenarios of what can happen when division is successful:
                # rla: Feb|Mar 2024
                #### WITH MULTIPLE SYMBIONTS PER CELL ####
                - both parent and child find happy cell homes
                    if cell is below max capacity:
                    - parent and child both stay in current cell
                    if cell is above max capacity:
                    - parent may stay and child may move into new cell
                    - child may stay and parent may move into new cell
                    - note -- could be that an open cell is outside the scope
                      of our 2D grid, if the parent lives on the top or bottom
                      border
                if cell is at max capacity and no open neighboring cells:
                - no room adjacent, so one of the parent or child is evicted
                ###########################################
            This will also update the next end-of-G0 time for the parent,
            should that be able to occur (or scheduling digestion or eviction
            prior should the parent not be able to make it to the next
            end-of-G0).
        Parameters:
            current_time: the current simulation time -- @ end of G1SG2M (float)
        Returns:
            a list containing the status resulting from the mitosis, and the
            resulting child symbiont; possible statuses returned:
                SymbiontState.PARENT_INFECTS_OUTSIDE
                SymbiontState.CHILD_INFECTS_OUTSIDE
                SymbiontState.BOTH_STAY
                SymbiontState.PARENT_NO_AFFINITY (division affinity)
                SymbiontState.CHILD_NO_AFFINITY (division affinity)
                SymbiontState.PARENT_EVICTED
                SymbiontState.CHILD_EVICTED
        '''

        self._time_of_next_end_g1sg2m = INFINITY
        return_status_and_child = None
    
        assert(self._prev_event_type == EventType.END_G0)

        # bgl: Feb|Mar 2024
        assert(self._getState() == SymbiontState.IN_G1SG2M)

        # bgl: Feb|Mar 2024
        # when processing _any_ event, need to update the photosynthetic
        # production projection for _all_ symbionts in that cell (each had
        # already projected to ensure getting at least to this event, so update
        # the bank of each symbiont in the current cell)
        self._projectSurplusForAll(current_time, update_bank = True)

        # also note that in endOfG0(), we precomputed what the additional cost of
        # division would entail, and if more than symbiont can afford, would never
        # have gotten here, but would have escaped or been digested earlier
        #
        # bgl: Feb|Mar 2024
        assert(self._getState() == SymbiontState.IN_G1SG2M)
        # rla: Feb|Mar 2024 
        # [surplus_at_end, time_of_digestion, time_of_exit] = \
        #    self._computeSurplusAtEventEnd(self._prev_event_time, current_time, \
        #                cell_symbionts, roll_back_others = False)
        # [surplus_at_end, time_of_digestion, time_of_exit] = \
        #     self._computeSurplusAtEventEnd(self._prev_event_time, current_time, \
        #                 cell_symbionts)
        # self._computeSurplusAtEventEnd(self._prev_event_time, current_time, \
        #                              state = SymbiontState.IN_G1SG2M)
    
        # assert(time_of_digestion == None) # should not get here otherwise...
        # assert(time_of_exit == None)      # should not get here otherwise...
    
        self._num_divisions = self._num_divisions + 1
    
        # try to find an open cell for the new symbiont (note: it may be
        # a (modeled) cell outside our environment if the original symbiont is
        # in the top or bottom row of the host cell grid)
        # bgl: Feb|Mar 2024
        # (add logic to allow for enforcing matching clade in multi-occupied cell)
        open_cell = Symbiont.sponge.checkForOpenAdjacentCell(self._cell, self._clade_number)
        #open_cell = Symbiont.sponge.checkForOpenAdjacentCell(self._cell)
    
        # self._cell is the parent's cell
        # inside _SymbiontCopy, self refers to parent and new_symbiont to child
        # To do:
        # - if there is an open cell (either inside or outside), flip fair coin
        #       to determine who stays and who goes
        #   - cases:
        #       (a) open cell outside environment:
        #           child goes (no Cell to SymCopy), parent stays (no change to self._cell)
        #           child stays (self._cell to SymCopy), parent goes (self._cell = no Cell)
        #       (b) open cell inside the environment:
        #           child goes (open_cell to SymCopy), parent stays (no change to self._cell)
        #           child stays (self._cell to SymCopy), parent goes (self._cell = open_cell)
        # - if no open cell, flip fair coin to determine who stays
        #       (already modeled below -- use as an example)
        #
        # - for event scheduling:
        #       - update parent's next time of endG0 only if parent is not evicted
        #             or if parent is not moving to cell outside our grid

        #########################################################################
        if open_cell == self._cell:
        #########################################################################
            ## both parent and child stay in current cell ##
            child = self._SymbiontCopy(self._cell, current_time) # creating child
            self._cell.addSymbiont(child)   # update current cell to contain child
            return_status_and_child = [SymbiontState.BOTH_STAY, child]
    
        #########################################################################
        elif open_cell == SymbiontState.CELL_OUTSIDE_ENVIRONMENT:
        #########################################################################
            # this is not an eviction -- we are presuming the new symbiont is
            # infecting a cell outside the scope of our modeled environment
            # (above top row or below bottom row); 
            # child is created but it or parent presumed gone outside our
            # environment -- call the _SymbiontCopy method
            prob = RNG.uniform(0, 1, Stream.EVICTION)
            if prob < self._my_clade.getParentEvictionProb():
                ######################################################
                ## child stays in current cell, parent infects outside
                ######################################################
                #print(f"Parent infecting cell along border {self._id}")
                child = self._SymbiontCopy(self._cell, current_time)
                # bgl: Feb|Mar 2024
                self._cell.removeSymbiont(self) # remove parent from current cell
                self._cell.addSymbiont(child)   # update current cell to contain child
                #self._cell.setSymbiont(child, current_time) # update cell to contain child
                self._cell = None  # parent infects outside (i.e., no cell in model)
                return_status_and_child = [SymbiontState.PARENT_INFECTS_OUTSIDE, child]
                # info on current cell inhabited by child occurs in _SymbiontCopy
            else:
                ######################################################
                ## parent stays in current cell, child infects outside
                ######################################################
                #print(f"Child infecting cell along border {self._id}")
                no_cell = None
                child = self._SymbiontCopy(no_cell, current_time) 
                return_status_and_child = [SymbiontState.CHILD_INFECTS_OUTSIDE, child]
                # no need to update new cells inhabited for either parent or child
            #
        #########################################################################
        elif open_cell is not None:  # there is an open cell for child or parent
        #########################################################################
            # if open_cell is this symbiont's cell -- rollback
            # else VV

            prob = RNG.uniform(0, 1, Stream.EVICTION)
            if prob < self._my_clade.getParentEvictionProb():
                ## child stays in current cell, parent moves to the new open cell
                #print(f"Parent goes to open cell {self._id}")
                child = self._SymbiontCopy(self._cell, current_time)
                # bgl: Feb|Mar 2024
                self._cell.removeSymbiont(self)  # remove parent from current cell
                self._cell.addSymbiont(child)    # current cell now contains child
                #self._cell.setSymbiont(child, current_time) # current cell now contains child
    
                # use affinity values to determine if symbiont is phagocytosed
                phagocytosed = Symbiont._determinePhagocytosis(self._my_clade, is_arrival = False)
                if phagocytosed:
                    self._cell = open_cell
                    # bgl: Feb|Mar 2024
                    open_cell.addSymbiont(self)  # open cell now contains parent
                    #open_cell.setSymbiont(self, current_time)  # open cell now contains parent
                    # info on new cell inhabited by child occurs in _SymbiontCopy;
                    # but need to record info on parent switching to new open cell
                    self._cells_inhabited.append(str(open_cell.getLevelRowCol()).replace(', ',','))
                    self._inhabit_times.append(current_time)
                    self._hcds_of_cells_inhabited.append(open_cell.getDemand())
                    return_status_and_child = [SymbiontState.BOTH_STAY, child]
                else:
                    # similar to parent evicted
                    self._cell = None # parent now homeless
                    return_status_and_child = [SymbiontState.PARENT_NO_AFFINITY, child]
            else:
                ## parent stays in current cell, child moves to the new open cell
                ## call symbiont copy constructor, place into open cell
                #print(f"Child goes to open cell {child.id}")
            
                # use affinity values to determine if symbiont is phagocytosed
                phagocytosed = Symbiont._determinePhagocytosis(self._my_clade, is_arrival = False)
                if phagocytosed:
                    # info on new cell inhabited by child occurs in _SymbiontCopy
                    child = self._SymbiontCopy(open_cell, current_time)
                    # bgl: Feb|Mar 2024
                    open_cell.addSymbiont(child)  # open cell now contains child
                    #open_cell.setSymbiont(child, current_time)
                    return_status_and_child = [SymbiontState.BOTH_STAY, child]
                    # no need to update new cells inhabited for either parent or child
                else:
                    no_cell = None
                    child = self._SymbiontCopy(no_cell, current_time)
                    return_status_and_child = [SymbiontState.CHILD_NO_AFFINITY, child]
            # 
        #########################################################################
        else: # there is no open cell for a new symbiont
        #########################################################################
            # rla: Feb|Mar 2024
            # if both stay in current cell, pass!
            if return_status_and_child != None:
                pass
            # there is no room for an additional symbiont, so flip a coin; 
            # if 0, the child would have been created but ejected into the pool;
            #     still, call _SymbiontCopy to make sure photosynthate is evenly
            #     divided -- parents stays put;
            # if 1, the parent will be evicted into the pool, and the copied 
            #     child will go into the current cell
            prob = RNG.uniform(0, 1, Stream.EVICTION) 
            if prob < self._my_clade.getParentEvictionProb():
                ## child stays in current cell, parent evicted into pool;
                ## call symbiont copy constructor, place child into current cell
                #print(f"Parent evicted into pool {self._id}")
                child = self._SymbiontCopy(self._cell, current_time)
                # bgl: Feb|Mar 2024
                self._cell.removeSymbiont(self) # remove parent from current cell
                self._cell.addSymbiont(child)   # current cell now contains child
                #self._cell.setSymbiont(child, current_time)  # old way just overwrote
                ##################
                self._cell = None  # parent now homeless
                return_status_and_child = [SymbiontState.PARENT_EVICTED, child]
                # info on new cell inhabited by child occurs in _SymbiontCopy
            else:
                ## parent stays in current cell, child evicted into pool;
                # child is created but presumed gone into the pool, so no cell
                #print(f"Child evicted into pool {child.id}")
                no_cell = None
                child = self._SymbiontCopy(no_cell, current_time)
                return_status_and_child = [SymbiontState.CHILD_EVICTED, child]
                # no need to update new cells inhabited for either parent or child
        #########################################################################
    
        # only in the parent-evicted or parent-infects-outside cases above 
        # do we NOT try to update the parent's next end of G0
        if not (return_status_and_child[0] == SymbiontState.PARENT_EVICTED or \
                return_status_and_child[0] == SymbiontState.PARENT_INFECTS_OUTSIDE or \
                return_status_and_child[0] == SymbiontState.PARENT_NO_AFFINITY):  
            # must make sure that parent can make it through this next G0 event
            # (e.g., could be producing at rate less than host cell demand, but
            # banked photosynthate is sufficient to allow it through a few events)
            time_of_end_of_g0 = self._computeNextEndOfG0(current_time)
            # note the parent's photosynthate has already been divided in _SymbiontCopy
            # bgl: Feb|Mar 2024
            # NB: not sure yet whether I need to roll back others in the event
            #   of eviction
            #[surplus_at_end, time_of_digestion, time_of_exit] = \
            #    self._computeSurplusAtEventEnd(current_time, time_of_end_of_g0, \
            #            cell_symbionts)
            # [surplus_at_end, time_of_digestion, time_of_exit] = \
            #     self._computeSurplusAtEventEnd(current_time, time_of_end_of_g0, \
            #             cell_symbionts, roll_back_others = True or False)
            # self._computeSurplusAtEventEnd(current_time, time_of_end_of_g0, \
            #                                 SymbiontState.IN_G0)
          
        # rla: Feb|Mar 2024
        # _projectToSymbiontNextEvent will project out across the event time
        # and if the symbiont cannot successfully complete, will schedule a
        # digestion or escape event; if can complete, will schedule end of
        # G0 for this symbiont
        print('##################### BEFORE')
        # following logic/ordering from endofG0; no longer want to compute
        # energetic demand of mitosis
        self._prev_event_time = current_time
        self._prev_event_type = EventType.END_G1SG2M

        self._projectToSymbiontNextEvent(
                               current_time    = current_time,
                               next_event_time = time_of_end_of_g0,
                               next_event_type = EventType.END_G0)
        print('##################### AFTER')
    
        # whatever the event might be, set the next event to occur for this symbiont
        # rla: Feb|Mar 2024
        # self._prev_event_time = current_time
        # self._prev_event_type = EventType.END_G1SG2M
        self._setNextEvent()
        # rla: Feb|Mar 2024
        self._cell.updateNextEventTime()

        return return_status_and_child
    
    #############################################################################
    def _symbiontExitFromCell(self, current_time: float) -> None:
        ''' method to handle cancelling of events in the event that a symbiont
            exits a host cell via digestion, escape, or denouement
        Parameters:
            current_time: current simulation time -- time of digestion (float)
        '''
        # rla: Feb|Mar 2024
        from simulation import Simulation
        cell_symbionts = self._cell.getSymbionts()

        for s in cell_symbionts:
            s_next_event_time, s_next_event_type = s.getNextSymbiontEvent()
            [surplus_at_end, time_of_digestion, time_of_exit] = \
                self._computeSurplusAtEventEnd(current_time, s_next_event_time, \
                    cell_symbionts)
            
            if surplus_at_end < 0:
                # first, handle end of G0 or G1SG2M events
                if s_next_event_type in [EventType.END_G0, EventType.END_G1SG2M]:
                    # cancel G0 or G1SG2M event
                    s.cancelEventWithinSymbiont(s_next_event_type)
                    Simulation.cancelFromEventList(s_next_event_time, s_next_event_type, s.getID())
                    # set time of digestion or escape
                    self._time_of_digestion = time_of_digestion
                    if time_of_exit is not None:
                        assert(time_of_digestion == INFINITY)
                        self._time_of_escape = time_of_exit
                    # set new event -- escape or digestion
                    s._setNextEvent()
                    s._cell.updateNextEventTime()

                # next, handle digestion/escape events
                elif s_next_event_type in [EventType.DIGESTION, EventType.ESCAPE]:
                    updated_event_time = time_of_digestion
                    if time_of_exit is not None:
                        assert(time_of_digestion == INFINITY)
                        updated_event_time = time_of_exit
                    # first update time of event within symbiont
                    s._next_event_time = updated_event_time
                    # we are assuming that a digestion/escape that must be moved earlier
                    # due to recomputing photosynthate surplus in response to a symbiont
                    # exit from the cell will remain the same event type, even if 
                    # _computeSurplusAtEventEnd above were to choose the opposite event type
                    if s_next_event_type == EventType.DIGESTION:
                        s._time_of_digestion = updated_event_time
                    else:
                        s._time_of_escape = updated_event_time
                    # then update time of event within EventList
                    EventList.changeEventTime(s_next_event_time, updated_event_time, s_next_event_type, s.getID())

    #############################################################################
    def digestion(self, current_time: float) -> None:
        ''' method to handle digestion of the current symbiont, simply
            removing the symbiont from its current host cell
        Parameters:
            current_time: current simulation time -- time of digestion (float)
        '''
        # bgl|rla: Feb|Mar 2024
        # when processing _any_ event, need to update the photosynthetic
        # production projection for _all_ symbionts in that cell (each had
        # already projected to ensure getting at least to this event, so update
        # the bank of each symbiont in the current cell)
        self._projectSurplusForAll(current_time, update_bank = True)

        # 8 Oct 2016: keep track of photosynthate at end, even if leaving...
        # useful for the detailed csv per-symbiont tracking
        self._photosynthate_surplus = 0  # digested, so must be 0!
        self._cell.removeSymbiont(self)

        self._symbiontExitFromCell(current_time)

        #######################################################################
        # (Let's keep these notes here for future reference...)

        # For other event-handling algorithms (endOfG0, endOfG1SG2M), at the
        # _start_ of the event handling, we updated the bank of all symbionts
        # to the time of that event, before then processing whatever needed to
        # take place for the event-associated symbiont.
        #
        # We need to do a similar thing here, as all symbionts in the cell
        # should first have their bank updated to _now_, since they were all
        # sharing the load across that time (previous update time until now) 
        # -- including the symbiont being digested.
        #
        # Then, we can take care of removing the digested symbiont from the cell
        # before considering any event cancellation / rescheduling for the other
        # symbionts in the cell. 

        ########################################
        
        # On digestion, the current symbiont is departing -- we have already
        # removed it above from the cell's list of symbionts.  At this point,
        # we need to consider the impact on the remaining symbionts in the
        # cell (whose bank should have been updated to now, at the start of
        # handling this event -- see above).  For each of those symbionts:
        #   - If the symbiont had an end-of-G0 or end-of-G1SG2M scheduled,
        #     either there will be a new digestion/escape scheduled because
        #     of the reduction in load-sharing, or their G0/G1SG2M future
        #     event will remain albeit with a reduced amount of bank at that
        #     time.
        #   - If the symbiont had a digestion or escape scheduled, that event
        #     will surely be moved earlier because of the reduced load-sharing.

        ########################################

        # Rebecca and I determined that we don't need the additional load of
        # cancelling every single G0/G1SG2M/denoument event (as those times may
        # not change), but do/should cancel any scheduled digestion/escape since
        # that will be moved earlier in simulated time.
        #
        # We also noted that we may not need to actually cancel (in the event
        # list) the digestion/escape, but might just be able to get away with 
        # changing its time --- see the newly-added changeEventTime method
        # inside event_list.py.


    #############################################################################
    def escape(self, current_time: float) -> None:
        ''' method to handle the current symbiont escaping digestion, simply
            removing the symbiont from its current host cell (presumed to be
            going back to the pool)
        Parameters:
            current_time: current simulation time -- time of escape (float)
        '''
        # bgl|rla: Feb|Mar 2024
        # when processing _any_ event, need to update the photosynthetic
        # production projection for _all_ symbionts in that cell (each had
        # already projected to ensure getting at least to this event, so update
        # the bank of each symbiont in the current cell)
        self._projectSurplusForAll(current_time, update_bank = True)

        # 8 Oct 2016: keep track of photosynthate at end, even if leaving...
        # useful for the detailed csv per-symbiont tracking
        self._photosynthate_surplus = 0  # escaping, so must be 0!
        self._cell.removeSymbiont(self)

        self._symbiontExitFromCell(current_time)

    #############################################################################
    def denouement(self, current_time: float) -> None:
        ''' method to handle the current symbiont's denouement and transition 
            back to the pool, simply removing the symbiont from its current
            host cell (presumed to be going back to the pool) and appropriately
            computing the symbiont's photosynthate surplus at denouement (for
            tracking)
        Parameters:
            current_time: current simulation time -- time of denouement (float)
        '''
        # bgl|rla: Feb|Mar 2024
        # when processing _any_ event, need to update the photosynthetic
        # production projection for _all_ symbionts in that cell (each had
        # already projected to ensure getting at least to this event, so update
        # the bank of each symbiont in the current cell)
        self._projectSurplusForAll(current_time, update_bank = True)
        
        # 8 Oct 2016: keep track of photosynthate at end, even if leaving...
        # useful for the detailed csv per-symbiont tracking
        # (see comment in endOfG0 on use of computeSurplus method...)
        # rla: Feb|Mar 2024
        cell_symbionts = self._cell.getSymbionts()
        [surplus_at_end, time_of_digestion, time_of_exit] = \
            self._computeSurplusAtEventEnd(self._prev_event_time, current_time, \
                    cell_symbionts)
        # [surplus_at_end, time_of_digestion, time_of_exit] = \
        #     self._computeSurplusAtEventEnd(self._prev_event_time, current_time, \
        #                                   state = None)
        assert(time_of_digestion == None)  # sanity check
        assert(time_of_exit == None)       # sanity check
        assert(surplus_at_end >= 0)  # o/w, shouldn't have made it to denouement

        self._photosynthate_surplus = surplus_at_end
        self._cell.removeSymbiont(self)

        self._symbiontExitFromCell(current_time)

    #############################################################################
    @staticmethod
    def _determinePhagocytosis(clade: Clade, is_arrival: bool) -> bool:
        ''' static method to determine whether phagocytosis happens based
            on clade-specific probability
        Parameters:
            clade: Clade object corresponding to a symbiont clade
            is_arrival: True if symbiont is arriving, False o/w
        Returns:
            True if the symbiont will be phagocytosed (whether on arrival
            or division), False o/w
        '''
        #############################################################################
        # New method, as of spring 2019: each algal clade takes on a different 
        # arrival and division affinity "strategy", denoted by a number, which has
        # a different probability
        #    u <- fair coin flip using either 
        #           Stream.ARRIVAL_AFFINITY or Stream.DIVISION_AFFINITY
        #    p <- clade-specific probability of entry/maintenance based
        #           (provided by user in input file)
        #    if (u < p) {phagocytosed}  else  {not phagocytosed}
        #############################################################################
        if is_arrival:
            affinity_prob = RNG.uniform(0, 1, Stream.ARRIVAL_AFFINITY)  # flip a coin
            # retrieve arrival affinity probability based on clade
            clade_prob = clade.getArrivalAffinityProb()
        else:
            affinity_prob = RNG.uniform(0, 1, Stream.DIVISION_AFFINITY) # flip a coin
            # retrieve division affinity probability based on clade
            clade_prob = clade.getDivisionAffinityProb()

        # if u < p, return True (phagocytosed), o/w False (not phagocytosed)
        return affinity_prob < clade_prob

    #############################################################################
    # def eviction(self, current_time: float) -> None:
        # note there is no separate eviction event because we handle the eviction
        # within endOfG1SG2M() -- to compute stats appropriately, in the case of
        # parent eviction we need a reference to the cell, so we don't handle this
        # as a separate event

    #############################################################################
    # bgl|rla: Feb|Mar 2024
    #def getNextEvent(self) -> tuple[float, EventType]:
    def getNextSymbiontEvent(self) -> tuple[float, EventType]:
        ''' method to return the next event time and type for this symbiont,
            as stored internally in the symbiont
        Returns:
            a tuple containing the time of the next event and the event type
        '''
        return (self._next_event_time, self._next_event_type)

    #############################################################################
    def _setNextEvent(self) -> None:
        ''' method to update the symbiont's internal state keeping track of its
            next event time and type to occur
        '''
        # NOTE: the order of occurrence is important here -- an event higher
        # in the order takes precedence of an event lower in the order should
        # there be identical event times
        self._next_event_time = self._time_of_next_end_g0
        self._next_event_type = EventType.END_G0

        if self._time_of_next_end_g1sg2m < self._next_event_time:
            self._next_event_time = self._time_of_next_end_g1sg2m
            self._next_event_type = EventType.END_G1SG2M

        if self._time_of_escape < self._next_event_time:
            self._next_event_time = self._time_of_escape
            self._next_event_type = EventType.ESCAPE

        if self._time_of_digestion < self._next_event_time:
            self._next_event_time = self._time_of_digestion
            self._next_event_type = EventType.DIGESTION

        if self._time_of_denouement < self._next_event_time:
            self._next_event_time = self._time_of_denouement
            self._next_event_type = EventType.DENOUEMENT

    #############################################################################
    def _scheduleInitialEvents(self, current_time: float) -> None:
        ''' private method to set up the initial events for an arriving
            symbiont
        Parameters:
            current_time: time of the arrival
        Notes:
            - In the multi-symbiont-per-cell case, an arrival will require 
              recomputation of photosynthate production/demand for the
              symbionts who were already present in the cell --- since there
              will now be an additional symbiont who will be helping to carry
              the load of the host cell demand across time.
            - On arrival, we need to:
              (a) cancel any digestion/escape events for any of the
                  previously-occupying symbionts in this cell.
                    - The arrival of the new symbiont may preclude that
                      digestion/ escape schedule for a future time (i.e., the
                      sharing of demand may help the digesting/escaping
                      symbiont to instead persist).
                    - It may be the case that such digestion/escape may simply
                      be rescheduled for a later time, but it may be that the
                      digestion/ escape is altogether eliminated because of the
                      additional sharing of demand.
              (b) update the new minimum next event time for the cell based on
                  the updated events (which should now only consist of
                  end-of-G0 and end-of-G1SG2M events)
              (c) recompute [simply for the sake of projecting into the future 
                  for event scheduling] the photosynthate production & demand
                  for all symbionts in the cell, including the newly arriving
                  symbiont
        '''
        # bgl: Feb|Mar 2024
        ## FIRST, LOGIC FOR OTHER SYMBIONTS ALREADY IN THE CELL
        from simulation import Simulation

        # when processing _any_ event (including an arrival), need to update
        # the photosynthetic production projection for _all_ symbionts in that
        # cell (each had already projected to ensure getting at least to this
        # event, so update the bank of each symbiont in the current cell)
        self._projectSurplusForAll(current_time, update_bank = True, \
            arriving_symbiont = None)  # bgl: Feb|Mar 2024
            #arriving_symbiont = self)
            # ^^^ @ start of arrival, don't want to include this arriving
            #     symbiont as we should be updating all the other symbionts'
            #     bank to the current point -- the time of the arrival --
            #     across which time they should _not_ be dividing using the
            #     just-arrived symbiont
        print('########### JUST UPDATED ALL SYMBIONTS BANK ON NEW ARRIVAL')

        # cancel any digestion/escape events for any of the previously
        # occupying symbionts
        # NB: the newly-arriving symbiont has not yet been added to the cell's
        #     symbiont list at this point; that is handled once __init__ [which
        #     called this method] returns to generateArrival()
        cell_symbionts = self._cell.getSymbionts()  # returns a copy
        for s in cell_symbionts:
            s_next_event_time, s_next_event_type = s.getNextSymbiontEvent()
            # because this corresponds to an arrival, only a digestion or escape
            # are the events that might be changed and/or precluded because of
            # the additional shared help in host cell demand
            if s_next_event_type in [EventType.DIGESTION, EventType.ESCAPE]:
                # cancel that event, which will also update the symbiont's
                # next scheduled event time appropriately
                s.cancelEventWithinSymbiont(s_next_event_type)
                # also cancel from the simulation event list
                Simulation.cancelFromEventList(s_next_event_time, s_next_event_type, s.getID())

        # sanity check -- there should be no digestion or escape events
        # scheduled for _any_ symbiont in this cell at this point
        for s in cell_symbionts:
            assert(s._time_of_escape == INFINITY)
            assert(s._time_of_digestion == INFINITY)

        ## NOW LOGIC FOR THIS ARRIVING SYMBIONT

        # set up the long-term denouement time (which we presume is because the
        # symbiont leaves the host cell in order to reproduce) -- this may be
        # superseded by the exit strategy computation below...

        m = self._my_clade.getAvgResidenceTime()
        f = self._my_clade.getResidenceFuzz()
        residence_time = RNG.fuzz(m, f, Stream.TIME_DENOUEMENT)
        self._time_of_denouement = current_time + residence_time
        #print(f">>> RESTIME: {residence_time}")

        # set up the next mitosis to be on schedule, +/- fudge factor;
        # this allows us to model "accumulated bad decisions" RE division;
        # mitosis is scheduled using two events (see 17 Mar 2015 email with
        # photo of board notes):
        #     1) end of G0 / start of G1,S,G2,M
        #     2) end of M / start of G0
        # in this way, if the dividing cell's photosynthate surplus goes
        # negative at any time during the G1,S,G2,M phase -- when the additional
        # subtraction due to cost of mitosis occurs -- that cell will either
        # be digested or exit-expulsed, and the division never occurs in our 
        # model (it could be the case that if exit-expulsed, the cell would 
        # divide elsewhere, but that is beyond our model scope);
        # 
        # moreover, with respect to the times for the G0,G1,S,G2,M process, we
        # presume that _only_ G0 will vary for hi vs med vs low reproducers;
        # the average time for G1,S,G2,M will be the same regardless -- in other
        # words low reproducers will spend more time in G0 banking photosynthate
        # before entering the "going to divide" state of G1,S,G2,M from which there
        # is no turning back once committed
        self._time_of_next_end_g0 = self._computeNextEndOfG0(current_time)

        # if this symbiont can't produce photosynthate at a rate sufficient
        # to meet the host cell's demand through G0, schedule the exit strategy...

        # bgl: Feb|Mar 2024
        # update the cell's new common minimum event time, which we want to have
        # include this arriving cell's times (so call symbiont's setNextEvent first --
        # which will be called again in generateArrival, perhaps with a new dig/esc
        # event thanks to the logic below)
        self._setNextEvent()
        self._cell.updateNextEventTime(arriving_symbiont = self)
        cell_next_event_time = self._cell.getNextEventTime()
        assert(cell_next_event_time is not None)

        assert(self._getState() == SymbiontState.IN_G0)
        assert(self._cell is not None)
        assert(self not in cell_symbionts)  # this method is called @ sim start

        # recompute the future production for every symbiont in the cell,
        # including the newly-arrived one (which will be officially added in
        # generateArrival), since the newly-arrived symbiont may have affected
        # the future viability of previously existing symbionts due to shared
        # host cell demand;  set arriving_symbiont to self, reflecting that
        # the arriving symbiont has not yet been added to the cell's list of
        # symbionts for sharing the HCD load
        self._projectSurplusForAll(cell_next_event_time, \
            update_events = True, update_bank = False, arriving_symbiont = self)
        print('########### JUST REUPDATED ALL SYMBIONT EVENTS ON NEW ARRIVAL')

    #############################################################################
    # bgl: Feb|Mar 2024
    #def _computeProductionRate(self, is_copy: bool, current_time: float) -> float:
    def _computeProductionRate(self, is_copy: bool) -> float:
        ''' computes the photosynthetic production rate (PPR) of this symbiont
        Parameters:
            is_copy: True if this symbiont is arriving via mitosis; False o/w
        '''
        # photosynthetic production rate is a function of the clade (corresponding
        # rates defined in Parameters) and host cell location, with production rate
        # decreasing linearly from north to south (moving away from the sun);

        # for clade X:
        # for now presume that rho_X decreases linearly from full rho_X at the
        # topmost row to rho_X / k at the lowest row; hence, the endpoints of the
        # corresponding line are (rho_X, 0) and (rho_X / k, N-1), where N is the
        # maximum number of rows;  after dervation, this should give a line
        # equation of 
        #           y = rho_X + ((1-k)/k)*(x*rho_X/(N-1))
        # where y is the production rate and x is the corresponding row
        k = self._my_clade.getPhotosyntheticReduction()
        #######################################################################
        if is_copy:
            rho = self._production_rate  # this is an exact copy of parent
        else:
            rho = self._my_clade.getPPR()
        #######################################################################
        #NOTE: MAY NEED TO CHANGE -- to address change in rate across levels
        num_levels, num_rows, num_cols = Symbiont.sponge.getDimensions()
        level, row, col = self._cell.getLevelRowCol() # row x in the equation above
        rate = rho + (float(1-k)/k) * (row*rho/float(num_rows-1))

        ## 12 Apr 2016
        # for fuzzing, use normal with  95% of the data b/w (mu +/- mu*f) -- 
        # see implementation in rng.py; here there can be different fuzzing
        # parameters depending on whether this is an "original" symbiont or a
        # copy via division
        if is_copy:
            # use multi-distro division fuzzing -- see implementation in rng.py
            [fuzz_amt, mutation] = RNG.divfuzz(rate, self._my_clade, Stream.PHOTOPROD_MUTATION)
            fuzzed_rate = rate
            if mutation == MutationType.DELETERIOUS:
                fuzzed_rate -= fuzz_amt  # deleterious photoprod reduces
            else: # mutation == MutationType.BENEFICIAL:
                fuzzed_rate += fuzz_amt  # beneficial photoprod increases
            # uncomment below if want to see info about mutations...
            '''
            if mutation != MutationType.NO_MUTATION:
                mut_type = 'DEL' if mutation == MutationType.DELETERIOUS else 'BEN'
                print(f'@ t={current_time} {self._id} {mut_type} ppr: {rate} {fuzzed_rate}")
            '''
        else:
            # use normal -- see implementation in rng.py
            m = rate
            f = self._my_clade.getPPRFuzz()
            fuzzed_rate = RNG.fuzz(m, f, Stream.PHOTOPROD)

        return fuzzed_rate  # y in the equation above

    #############################################################################
    ''' simple getter/accessor methods '''
    def getID(self)            -> int:       return self._id
    def getCladeNumber(self)   -> int:       return self._clade_number
    def getArrivalTime(self)   -> float:     return self._arrival_time
    def getPrevEventType(self) -> EventType: return self._prev_event_type

    #############################################################################
    def __str__(self) -> str:
        ''' create a useful string for printing a symbiont
        Returns:
            str representation of this Symbiont object
        '''
        level, row, col = self._cell.getLevelRowCol() if self._cell is not None else [-1,-1,-1]
        string = f'Symbiont {self._id}' \
            + f': @({level}{row},{col})\n\t' \
            + f'clade        : {self._clade_number}\n\t' \
            + f'cell         : {self._cell}\n\t' \
            + f'prod rate    : {self._production_rate}\n\t' \
            + f'surplus      : {self._photosynthate_surplus}\n\n\t' \
            + f'next event time  : {self._next_event_time}\n\t' \
            + f'next event type  : {self._next_event_type.name}\n\t' \
            + f'arrival time     : {self._arrival_time}\n\t' \
            + f'last event time  : {self._prev_event_time}\n\t' \
            + f'time of end-G0   : {self._time_of_next_end_g0}\n\t' \
            + f'time of end-G1->M: {self._time_of_next_end_g1sg2m}\n\t' \
            + f'time of exit exp : {self._time_of_escape}\n\t' \
            + f'time of digest   : {self._time_of_digestion}\n\t' \
            + f'time of res exp  : {self._time_of_denouement}\n\t' \
            + f'prev update time : {self._prev_update_time}'
            # bgl: Feb|Mar 2024
        return string

    ###################################################################################
    def csvOutputOnExit(self, current_time: float, exit_status: SymbiontState) -> None:
        ''' method to write per-symbiont information to CSV file (if requested)
        Parameters:
            current_time: current simulation time (float)
            exit_status: exit status of symbiont (one from SymbiontState enumeration)
        '''
        # Note there are four ways for a symbiont to exit the system:
        #   (1) residence expulsion -- just ends its natural time in the sponge
        #   (2) digested (during mitosis or not)
        #   (3) escape (during mitosis or not)
        #   (4) evicted (could be the parent or the child)
        # This method should be called whenever one of those happens, giving CSV
        # output for the statistics of that exiting symbiont.
        #
        if not Symbiont._write_csv: return
        strval  = str(self._id)                + ','    # overall symbiont id number
        strval += str(self._how_arrived.name)  + ','    # via pool or division
        strval += str(self._parent_id)         + ','    # id of parent, -1 via pool
        strval += str(self._agent_zero)        + ','    # id of ultimate ancestor
        strval += str(self._clade_number)      + ','
        strval += str(self._mitotic_cost_rate) + ','
        strval += str(self._production_rate)   + ','
        strval += str(self._arrival_time)      + ','    # arrival time
        strval += str(current_time)            + ','    # exit time (1 of 4 above)
        strval += str(exit_status.name)        + ','
        ## begin added 31 Oct 2016
        strval += str(self._prev_event_time)             + ','
        strval += str(self._prev_event_type.name)        + ',' 
        ## end added 31 Oct 2016
        strval += str(current_time - self._arrival_time) + ','  # residence time
        strval += str(self._surplus_on_arrival)          + ','  # surplus @ arrival
        strval += str(self._photosynthate_surplus)       + ','  # surplus @ exit
        strval += str(self._num_divisions)               + ','  # num successful divs
        strval += str(self._time_of_escape)              + ','
        strval += str(self._time_of_digestion)           + ','
        strval += str(self._time_of_denouement)          + ','
        if exit_status == SymbiontState.STILL_IN_RESIDENCE:
            strval += SymbiontState.STILL_IN_RESIDENCE.name + ','
        else:
            strval += "NOT_IN_RESIDENCE,"
        # append cells (perhaps multiple) inhabited by symbiont -- separate w/ ;
        for i in range(len(self._cells_inhabited)):
            strval += ('"' if i == 0 else ';') + str(self._cells_inhabited[i])
        if len(self._cells_inhabited) > 0: strval += '"'
        # append times (perhaps multiple) cells were inhabited by symbiont
        strval += ','
        for i in range(len(self._inhabit_times)):
            strval += ('' if i == 0 else ';') + str(self._inhabit_times[i])
        # append hcds (perhaps multiple) of cells inhabited by symbiont
        strval += ','
        for i in range(len(self._hcds_of_cells_inhabited)):
            strval += ('' if i == 0 else ';') + str(self._hcds_of_cells_inhabited[i])
        # append g0 times symbiont experienced; separate diff times by semicolon
        strval += ','
        # if symbiont is a child immediately evicted or child who infected outside
        # that means it received a g0 time that was never used -- let's not
        # include the g0 time in the output; a parent who finished g1sg2m but was 
        # evicted or infected outside does not get assigned a new g0 time -- see 
        # endG1SG2M(); for a symbiont who is digested or escapes during G0 (added 
        # via _computeNextEndOfG0() near end of endG1SG2M()), want to keep that g0
        # time in the output as it was "partially" used...  (see overall comments
        # appended to the bottom of this program)
        len_g0_times = len(self._g0_times)
        if exit_status == SymbiontState.CHILD_INFECTS_OUTSIDE or \
           exit_status == SymbiontState.CHILD_EVICTED: len_g0_times -= 1
        for i in range(len_g0_times):
            strval += ('' if i == 0 else ';') + str(self._g0_times[i])
        #
        # append g1sg2m times symbiont experienced; separate diff times by semicolon
        strval += ','
        for i in range(len(self._g1sg2m_times)):
            strval += ('' if i == 0 else ';') + str(self._g1sg2m_times[i])
        #
        # append cnt of open cells (at division) seen; separate diff times by semicolon
        #strval += ','
        #for i in range(len(self._cells_at_division)):
        #    strval += ('' if i == 0 else ';') + str(self._cells_at_division[i])
        #
        strval += '\n'
        Symbiont._csv_file.write(strval)
        Symbiont._csv_writes += 1

    #############################################################################
    @classmethod
    def computeCumulativeCladeProportions(cls) -> None:
        ''' class-level method to sets up an array of cumulative proportions
            (probabilities) so that different clades can have different
            probabilities of arriving -- used when generating a symbiont
            arrival
        '''
        cls.clade_cumulative_proportions = numpy.cumsum(Parameters.CLADE_PROPORTIONS)
        # set last entry to 1.0 just to be safe (avoid roundoff errors)
        cls.clade_cumulative_proportions[-1] = 1.0
        if len(cls.clade_cumulative_proportions) != Parameters.NUM_CLADES:
            msg = "Unable to compute cumulative clade proportions -- please check " + \
                  "CLADE_PROPORTIONS entry in CSV input file"
            print(msg)
            sys.exit()

    @classmethod
    def openCSVFile(cls, csv_fname: str) -> None:
        cls._write_csv = True
        cls._csv_file = open(csv_fname, "w")
        cls._csv_file.write(\
           'symbID,poolOrDiv,parent,agentZero,clade,mcr,ppr,'\
          +'arrTime,exitTime,exitStatus,lastEventTime,lastEventType,'\
          +'resTime,arrSurplus,exitSurplus,divs,'\
          +'tEsc,tDig,tRes,stillInRes,cells,inhabitTimes,'\
          +'hcds,g0Times,g1sg2mTimes,cellsAtDiv\n');

    @classmethod
    def csvOutputAtEnd(cls, current_time: float) -> None:
        ''' class-level method to write to CSV the per-symbiont information (if
            requested) at the end of the simulation
        Parameters:
            current_time: current simulation time @ end (float)
        '''
        if not cls._write_csv: return
        # write output for all those still in residence
        rows, cols = cls.sponge.getDimensions()
        for r in range(rows):
            for c in range(cols):
                cell = cls.sponge.getCell(r,c)
                symbiont = cell.getSymbiont()
                if symbiont is not None:
                    symbiont.csvOutputOnExit(current_time, SymbiontState.STILL_IN_RESIDENCE)
        Symbiont._csv_file.close()
  

    #############################################################################
    # bgl: Feb|Mar 2024
    @classmethod
    def findOpenCell(cls, which_clade: int) -> Cell | None:
    #def findOpenCell(cls) -> Cell:
        ''' static method to find and select an open cell at random among all
            available cells in the entire Sponge grid;  a cell is available if
            it is entirely unoccupied or if there is space within the cell and
            the current occupants are of the same clade
        Parameters:
            which_clade: the (integer) clade to which the ultimately-occupying
                symbiont belongs -- for allowing multiple of same type only
                within the same cell
        Returns:
            the Cell object selected, or None if none available for this clade
        '''
        # create a list of all open cells
        open_cells = []
        for l in range(Parameters.NUM_LEVELS):
            for r in range(Parameters.NUM_ROWS):
                for c in range(Parameters.NUM_COLS):
                    cell = cls.sponge.getCell(l,r,c)
                    # bgl: Feb|Mar 2024
                    if cell.isEmpty() or cell.isRoomFor(which_clade):
                    #if not cell.isOccupied():
                        open_cells.append(cell)

        # we should never call this method unless there is at least one open cell
        # bgl: Feb|Mar 2024 NOTE: no longer the case with multiple symbionts per
        #   cell, since there could open space but in a cell already claimed
        #   (occupied) by a different clade
        #assert(len(open_cells) > 0)  # sanity check
        if len(open_cells) == 0:
            return None

        # if any available, pick one @ random
        which = RNG.randint(0, len(open_cells)-1, Stream.OPEN_CELL_ON_ARRIVAL)
        return open_cells[which]

    #######################################################################################
    # bgl: Feb|Mar 2024
    #def findOpenCellWithin(cls, min_level: int, max_level: int,  
    #                            min_row: int, max_row: int, 
    #                            min_col: int, max_col: int) -> Cell:
    @classmethod
    def findOpenCellWithin(cls, min_level:   int, max_level: int,  
                                min_row:     int, max_row:   int, 
                                min_col:     int, max_col:   int,
                                which_clade: int ) -> Cell | None:
        ''' Static method to find and select an open cell at random among 
            available (unoccupied) cells within a particular section of the Sponge grid.
            Note that min_row and min_col are inclusive; max_row and max_col
            are exclusive.
        Paramters:
            min_level: the minimum level value to start the search (inclusive)
            max_level: the maximum level value to end the search (exclusive)
            min_row: the minimum row value to start the search (inclusive)
            max_row: the maximum row value to end the search (exclusive)
            min_col: the minimum col value to start the search (inclusive)
            max_col: the maximum col value to end the search (exclusive)
            which_clade: the (integer) clade to which the ultimately-occupying
                symbiont belongs -- for allowing multiple of same type only
                within the same cell
        Returns:
            the Cell object selected
        '''
        # create a list of all open cells
        open_cells = []

        for l in range(max_level - min_level):
            for r in range(max_row - min_row):
                for c in range(max_col - min_col):
                    cell = cls.sponge.getCell(min_level + l, min_row + r, min_col + c)
                    # bgl: Feb|Mar 2024
                    if cell.isEmpty() or cell.isRoomFor(which_clade):
                    #if not cell.isOccupied():
                        open_cells.append(cell)

        # we should never call this method unless there is at least one open cell
        # bgl: Feb|Mar 2024 NOTE: no longer the case with multiple symbionts per
        #   cell, since there could open space but in a cell already claimed
        #   (occupied) by a different clade
        #assert(len(open_cells) > 0)
        if len(open_cells) == 0:
            assert(False)  # we should never get here, as this method is only called @ sim start
            return None

        # if any available, pick one @ random
        which = RNG.randint(0, len(open_cells)-1, Stream.OPEN_CELL_ON_ARRIVAL)
        return open_cells[which]

    ################################################################################
    @classmethod
    def generateArrival(cls, 
                        current_time: float, 
                        num_symbionts: int,
                        CELL_REMOVE_ME: Cell = None,  # remove -- for testing only
                        ) -> Symbiont or None:
        ''' method to generate a symbiont arrival
        Parameters:
            current_time: time of the arrival (current simulation time) (float)
            num_symbionts: total number of symbionts
        Returns:
            a new Symbiont object, if the sponge is not already full; None o/w
        '''
        # no need to even try if there are no available cells
        # bgl/rla 11 dec 2023
        #if num_symbionts == Parameters.NUM_ROWS * Parameters.NUM_COLS:
        #if num_symbionts == Parameters.NUM_ROWS * Parameters.NUM_COLS * Parameters.NUM_LEVELS:

        # bgl: Feb|Mar 2024
        # NB: there still may not be space for a new arrival, e.g., if the only
        # open spots are cells that are already occupied by a clade not matching
        # the arriving clade
        if num_symbionts == cls.sponge.getMaxOccupancy():
            #logging.debug('\tNo cells available')
            return None

        # now handle the arrival -- pick a clade at random using the previously
        # defined cumulative proportions for clades...
        prob = RNG.uniform(0, 1, Stream.CLADE)
        clade = 0
        while prob >= cls.clade_cumulative_proportions[clade]: clade += 1
    
        # now determine if there is appropriate affinity for infection;
        # first grab the clade object and use it to calculate arrival affinity
        this_clade = Clade.getClade(clade)

        # if this symbiont has insufficient arrival affinity for host, can't get in
        phagocytosed = cls._determinePhagocytosis(this_clade, is_arrival = True)
        if phagocytosed:
            # symbiont has arrival affinity - find an open cell for this symbiont
            # bgl: Feb|Mar 2024
            open_cell = cls.findOpenCell(this_clade)
            if CELL_REMOVE_ME is not None: open_cell = CELL_REMOVE_ME
            print("!!!!!!!!!!! REMEMBER TO REMOVE CELL_REMOVE_ME PARAM AFTER TESTING !!!!!!!!")
            if open_cell is not None:
                # bgl: Feb|Mar 2024
                # when processing _any_ event, need to update the photosynthetic
                # production projection for _all_ symbionts in that cell  -- which
                # is handled by calling _projectSurplusForAll within 
                # _scheduleInitialEvents by way of Symbiont __init__
                symbiont = Symbiont(clade, open_cell, current_time)  
                open_cell.addSymbiont(symbiont)
            else:
                symbiont = None
            #open_cell = cls.findOpenCell()
            #symbiont  = Symbiont(clade, open_cell, current_time)  
            #open_cell.setSymbiont(symbiont, current_time)
        else:
            #logging.debug('\tNo affinity: clade %s' % (clade))
            symbiont = None

        return symbiont

######################################################################
# NOTES ON WHETHER LAST G0 TIME SHOULD BE EXCLUDED IN OUTPUT:
# (These notes are referenced in csvOutputOnExit() above)
# ####################################################################
# child infects outside
#     - does append new g0 in _scheduleInitialEvents
# child evicted
#     - does append new g0 in _scheduleInitialEvents
# parent evicted
#     - does not append new g0
# parent infects outside
#     - does not append new g0
# digestion during G0:     last event type is END_G1SG2M
#     - does append new g0 via _computeNextEndOfG0() near end of endG1SG2M()
# escape during G0:        last event type is END_G1SG2M
#     - does append new g0 via _computeNextEndOfG0() near end of endG1SG2M()
# digestion during G1SG2M: last event type is END_G0
#     - does not append a new g0 since doesn't get through G1SG2M
# escape during G1SG2M:    last event type is END_G0
#     - does not append a new g0 since doesn't get through G1SG2M
# denouement
#     - does not append new g0

################################################################################
def initialArrivalsTest() -> None:
    #####
    from parser import Parser
    from parameters import Parameters
    Parser.parseCSVInput("input.csv")
    Symbiont.computeCumulativeCladeProportions()
    RNG.initializeStreams()
    #####
    Parameters.NUM_LEVELS = 3;  Parameters.NUM_ROWS = 5; Parameters.NUM_COLS = 5
    sponge = Symbiont.sponge = \
        Sponge(num_levels = 3, num_rows = 5, num_cols = 5, \
               grid = GridType.SQUARE, max_per_cell = 5)
    cell = sponge.getCell(level = 2, row = 3, col = 4)
    #####
    s1 = Symbiont(clade_number = 0, cell = cell, current_time = 0.0)
    cell.addSymbiont(s1)
    print('-' * 40)
    s2 = Symbiont(clade_number = 0, cell = cell, current_time = 0.0)
    cell.addSymbiont(s2)
    print('-' * 40)
    s3 = Symbiont(clade_number = 0, cell = cell, current_time = 0.0)
    cell.addSymbiont(s3)
    print('-' * 40)
    s4 = Symbiont(clade_number = 0, cell = cell, current_time = 0.0)
    cell.addSymbiont(s4)    
    print('-' * 40)
    print(cell)


################################################################################
def insertAnArrivalTest() -> None:
    from event_list import EventList
    from parser import Parser

    Parser.parseCSVInput("input.csv")
    Symbiont.computeCumulativeCladeProportions()
    RNG.initializeStreams()


    #####
    event_list = EventList()
    Parameters.NUM_LEVELS = 3;  Parameters.NUM_ROWS = 5; Parameters.NUM_COLS = 5
    sponge = Symbiont.sponge = \
        Sponge(num_levels = 3, num_rows = 5, num_cols = 5, \
               grid = GridType.SQUARE, max_per_cell = 5)
    cell = sponge.getCell(level = 2, row = 3, col = 4)
    #####
    symbionts = [None] * 4
    for i in range(len(symbionts)):
        # "arrival" at time 0.0
        s = symbionts[i] = Symbiont(clade_number = 0, cell = cell, current_time = 0.0)
        next_time, next_type = s.getNextSymbiontEvent()
        new_event = Event(next_time, next_type, s)
        event_list.insertEvent(new_event)
        #
        cell.addSymbiont(s)
        print('-' * 40)

    import textwrap
    def printEventListAndSymbionts():
        print('>' * 80)
        print("\n\t".join(textwrap.wrap(event_list.__str__(), width=80)))
        print(f"cell._next_event_time = {symbionts[0]._cell.getNextEventTime()}")
        print('>' * 80)
        for symb in symbionts: 
            print(symb)
            print('\t' + '>'*40)
        print('>' * 80)

    printEventListAndSymbionts()

    ##############################
    ev = event_list.getNextEvent()
    t, type_, s = ev._time, ev._type, ev._symbiont
    print(t, type_, s._id)

    # THIS WILL BE END-OF-G0 FOR s3
    assert(s._id == 3)
    print(f'%%%%%%%%%% BEGIN END-OF-G0 FOR {s._id} %%%%%%%%%% ')
    s.endOfG0(t)
    print(f'%%%%%%%%%% END END-OF-G0 FOR {s._id} %%%%%%%%%% ')
    nt, ntype_ = s.getNextSymbiontEvent()
    new_event = Event(nt, ntype_, s)
    event_list.insertEvent(new_event)

    printEventListAndSymbionts()

    ##############################
    # ARRIVAL @ 12.42
    t = 12.42
    print(f'%%%%%%%%%% BEGIN ARRIVAL @ {t} %%%%%%%%%% ')
    num_symbionts = len(symbionts)
    # note the for-testing-only additional cell arg for now... remove later
    s = Symbiont.generateArrival(t, num_symbionts, cell)

    # sufficient affinity to infect, so set up next event for symbiont
    nt, ntype_ = s.getNextSymbiontEvent()
    new_event = Event(nt, ntype_, s)
    event_list.insertEvent(new_event)

    print(f'%%%%%%%%%% END ARRIVAL @ 12.42 %%%%%%%%%% ')

    printEventListAndSymbionts()

    ##############################
    ev = event_list.getNextEvent()
    t, type_, s = ev._time, ev._type, ev._symbiont
    print(t, type_, s._id)

    # THIS WILL BE END-OF-G0 FOR s2
    assert(s._id == 2)
    print(f'%%%%%%%%%% BEGIN END-OF-G0 FOR {s._id} %%%%%%%%%% ')
    s.endOfG0(t)
    print(f'%%%%%%%%%% END END-OF-G0 FOR {s._id} %%%%%%%%%% ')
    nt, ntype_ = s.getNextSymbiontEvent()
    new_event = Event(nt, ntype_, s)
    event_list.insertEvent(new_event)

    printEventListAndSymbionts()


################################################################################
def eventsTest() -> None:
    from event_list import EventList
    from parser import Parser
    from parameters import Parameters

    Parser.parseCSVInput("input.csv")
    Symbiont.computeCumulativeCladeProportions()
    RNG.initializeStreams()


    #####
    event_list = EventList()
    Parameters.NUM_LEVELS = 3;  Parameters.NUM_ROWS = 5; Parameters.NUM_COLS = 5
    sponge = Symbiont.sponge = \
        Sponge(num_levels = 3, num_rows = 5, num_cols = 5, \
               grid = GridType.SQUARE, max_per_cell = 5)
    cell = sponge.getCell(level = 2, row = 3, col = 4)
    #####
    symbionts = [None] * 4
    for i in range(len(symbionts)):
        # "arrival" at time 0.0
        s = symbionts[i] = Symbiont(clade_number = 0, cell = cell, current_time = 0.0)
        next_time, next_type = s.getNextSymbiontEvent()
        new_event = Event(next_time, next_type, s)
        event_list.insertEvent(new_event)
        #
        cell.addSymbiont(s)
        print('-' * 40)

    import textwrap
    def printEventListAndSymbionts():
        print('>' * 80)
        print("\n\t".join(textwrap.wrap(event_list.__str__(), width=80)))
        print(f"cell._next_event_time = {symbionts[0]._cell.getNextEventTime()}")
        print('>' * 80)
        for symb in symbionts: 
            print(symb)
            print('\t' + '>'*40)
        print('>' * 80)

    printEventListAndSymbionts()

    ##############################
    ev = event_list.getNextEvent()
    t, type_, s = ev._time, ev._type, ev._symbiont
    print(t, type_, s._id)

    # THIS WILL BE END-OF-G0 FOR s3
    assert(s._id == 3)
    print(f'%%%%%%%%%% BEGIN END-OF-G0 FOR {s._id} %%%%%%%%%% ')
    s.endOfG0(t)
    print(f'%%%%%%%%%% END END-OF-G0 FOR {s._id} %%%%%%%%%% ')
    nt, ntype_ = s.getNextSymbiontEvent()
    new_event = Event(nt, ntype_, s)
    event_list.insertEvent(new_event)

    printEventListAndSymbionts()

    ##############################
    ev = event_list.getNextEvent()
    t, type_, s = ev._time, ev._type, ev._symbiont
    print(t, type_, s._id)

    # THIS WILL BE END-OF-G0 FOR s2
    assert(s._id == 2)
    print(f'%%%%%%%%%% BEGIN END-OF-G0 FOR {s._id} %%%%%%%%%% ')
    s.endOfG0(t)
    print(f'%%%%%%%%%% END END-OF-G0 FOR {s._id} %%%%%%%%%% ')
    nt, ntype_ = s.getNextSymbiontEvent()
    new_event = Event(nt, ntype_, s)
    event_list.insertEvent(new_event)

    printEventListAndSymbionts()

    ##############################
    ev = event_list.getNextEvent()
    t, type_, s = ev._time, ev._type, ev._symbiont
    print(t, type_, s._id)

    # THIS WILL BE END-OF-G1SG2M FOR s3
    assert(s._id == 3)
    print(f'%%%%%%%%%% BEGIN END-OF-G1SG2M FOR {s._id} %%%%%%%%%% ')
    s.endOfG1SG2M(t)
    print(f'%%%%%%%%%% END END-OF-G1SG2M FOR {s._id} %%%%%%%%%% ')
    nt, ntype_ = s.getNextSymbiontEvent()
    new_event = Event(nt, ntype_, s)
    event_list.insertEvent(new_event)

    printEventListAndSymbionts()

    ##############################
    ev = event_list.getNextEvent()
    t, type_, s = ev._time, ev._type, ev._symbiont
    print(t, type_, s._id)

    # THIS WILL BE END-OF-G1SG2M FOR s2
    assert(s._id == 2)
    print(f'%%%%%%%%%% BEGIN END-OF-G1SG2M FOR {s._id} %%%%%%%%%% ')
    s.endOfG1SG2M(t)
    print(f'%%%%%%%%%% END END-OF-G1SG2M FOR {s._id} %%%%%%%%%% ')
    nt, ntype_ = s.getNextSymbiontEvent()
    new_event = Event(nt, ntype_, s)
    event_list.insertEvent(new_event)

    printEventListAndSymbionts()

################################################################################
def insertAnExitTest() -> None:
    from event_list import EventList
    from parser import Parser

    Parser.parseCSVInput("input.csv")
    Symbiont.computeCumulativeCladeProportions()
    RNG.initializeStreams()


    #####
    event_list = EventList()
    Parameters.NUM_LEVELS = 3;  Parameters.NUM_ROWS = 5; Parameters.NUM_COLS = 5
    sponge = Symbiont.sponge = \
        Sponge(num_levels = 3, num_rows = 5, num_cols = 5, \
               grid = GridType.SQUARE, max_per_cell = 5)
    cell = sponge.getCell(level = 2, row = 3, col = 4)
    #####
    symbionts = [None] * 4
    for i in range(len(symbionts)):
        # "arrival" at time 0.0
        s = symbionts[i] = Symbiont(clade_number = 0, cell = cell, current_time = 0.0)
        next_time, next_type = s.getNextSymbiontEvent()
        new_event = Event(next_time, next_type, s)
        print(new_event)
        event_list.insertEvent(new_event)
        #
        cell.addSymbiont(s)
        print('-' * 40)

    import textwrap
    def printEventListAndSymbionts():
        print('>' * 80)
        print("\n\t".join(textwrap.wrap(event_list.__str__(), width=80)))
        print(f"cell._next_event_time = {symbionts[0]._cell.getNextEventTime()}")
        print('>' * 80)
        for symb in symbionts: 
            print(symb)
            print('\t' + '>'*40)
        print('>' * 80)

    printEventListAndSymbionts()

    ##############################
    ev = event_list.getNextEvent()
    t, type_, s = ev._time, ev._type, ev._symbiont
    print(t, type_, s._id)

    # THIS WILL BE END-OF-G0 FOR s3
    assert(s._id == 3)
    print(f'%%%%%%%%%% BEGIN END-OF-G0 FOR {s._id} %%%%%%%%%% ')
    s.endOfG0(t)
    print(f'%%%%%%%%%% END END-OF-G0 FOR {s._id} %%%%%%%%%% ')
    nt, ntype_ = s.getNextSymbiontEvent()
    new_event = Event(nt, ntype_, s)
    event_list.insertEvent(new_event)

    printEventListAndSymbionts()
    
    s = symbionts[2]
    s._next_event_type = EventType.DIGESTION
    key = (s._next_event_time, EventType.END_G0, s._id)
    event = event_list._event_finder[key]
    event._type = EventType.DIGESTION
    event_list._event_finder.pop(key)
    key = (s._next_event_time, EventType.DIGESTION, s._id)
    event_list._event_finder[key] = event
    
    printEventListAndSymbionts()

    ##############################
    ev = event_list.getNextEvent()
    t, type_, s = ev._time, ev._type, ev._symbiont
    print(t, type_, s._id)

    # THIS WILL BE DIGESTION FOR s2
    assert(s._id == 2)
    print(f'%%%%%%%%%% BEGIN DIGESTION FOR {s._id} %%%%%%%%%% ')
    s.digestion(t)
    print(f'%%%%%%%%%% END DIGESTION FOR {s._id} %%%%%%%%%% ')
    nt, ntype_ = s.getNextSymbiontEvent()
    new_event = Event(nt, ntype_, s)
    event_list.insertEvent(new_event)

    printEventListAndSymbionts()


if __name__ == "__main__":
    #initialArrivalsTest()
    #eventsTest()
    insertAnExitTest()
