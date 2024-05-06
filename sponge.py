from __future__ import annotations
from parameters import Parameters
from rng_mt19937 import *
from enums import GridType, SymbiontState
import random

################################################################################
class Cell:
    ''' class to model a host cell, having a (row,col,level) position in a 2D grid
        of host cells and a photosynthetic demand
    '''
    __slots__ = ('_level', \
                 '_row', \
                 '_col', \
                 '_demand', \
                 # bgl: Feb|Mar 2024
                 #'_occupied', \
                 #'_symbiont', \
                 '_symbionts', \
                 # bgl: Feb|Mar 2024
                 #'_last_occupied_time', 
                 #'_sum_residence_time', 
                 #'_num_occupants',
                 '_max_occupants',
                 '_next_event_time' # for keeping all symbionts in this cell
                                    #   on the same synchronized clock for 
                                    #   computing shared load of host demand
                 )

    ###############################################
    # bgl: Feb|Mar 2024
    #def __init__(self, level: int, row: int, col: int) -> None:
    def __init__(self, level: int, row: int, col: int, max_occupants: int) -> None:
        ''' initializer method for a host cell object
        Parameters:
            level:         integer valued level number in [0,num_levels - 1]
            row:           integer valued row number in [0,num_rows - 1]
            col:           integer valued column number in [0,num_cols - 1]
            max_occupants: maximum number of symbionts this cell can hold
        '''
        self._level    : int   = level
        self._row      : int   = row
        self._col      : int   = col
        self._demand   : float = self.computeDemand()
        # bgl: Feb|Mar 2024
        #self._occupied : bool = False

        # bgl: Feb|Mar 2024
        #self._symbiont : 'Symbiont' = None   # null
        self._symbionts       : list[Symbiont] = []
        self._max_occupants   : int   = max_occupants
        self._next_event_time : float = None

        # used to track observation-persistent and time-persistent statistics of 
        # residence time per cell (and eventually, in simulation.py, per row);
        # Note that:
        #      _sum_residence_time / MAX_T  = prop. of time cell is occupied
        #      _sum_residence_time / # occupants = avg time per symbiont occupation
        # bgl: Feb|Mar 2024
        #self._last_occupied_time = INFINITY
        #self._sum_residence_time = 0
        #self._num_occupants = 0

    ################################
    ''' simple accessors/getters '''
    def getDemand(self)      -> float:               return self._demand
    def getLevelRowCol(self) -> tuple[int,int,int]:  return (self._level, self._row, self._col)

    # bgl: Feb|Mar 2024
    def getNumOccupants(self)  -> int:   return len(self._symbionts)
    def getMaxOccupants(self)  -> int:   return self._max_occupants
    def getNextEventTime(self) -> float: return self._next_event_time

    #######################################
    # bgl: Feb|Mar 2024
    def updateNextEventTime(self, arriving_symbiont: Symbiont = None) -> None:
        ''' updates this cell's next event time to be the minimum next event
            time among all symbionts in this cell -- called whenever a new
            symbiont is added to this cell or removed from this cell
        Parameters:
            arriving_sybmiont: if not None, corresponds to the now-arriving
                symbiont which has not yet been added to the cell's list of
                symbionts
        '''
        symbionts = self.getSymbionts()  # returns a copy
        if arriving_symbiont is not None:
            assert(arriving_symbiont not in symbionts)
            symbionts.append(arriving_symbiont)
        min_event_time = INFINITY
        for s in symbionts:
            s_next_time, s_next_type = s.getNextSymbiontEvent()
            if s_next_time < min_event_time:
                min_event_time = s_next_time
        self._next_event_time = min_event_time  # min event time for this cell (self)

    # bgl: Feb|Mar 2024
    # (for now, intentionally returning shallow rather than deep copy)
    #def getSymbiont(self) -> 'Symbiont' or None: return self._symbiont
    def getSymbionts(self) -> list[Symbiont]:     return self._symbionts.copy()

    # bgl: Feb|Mar 2024
    #def isOccupied(self)  -> bool: return self._occupied
    def isOccupied(self)  -> bool:  return len(self._symbionts) > 0
    def isEmpty(self)     -> bool:  return len(self._symbionts) == 0

    # bgl: Feb|Mar 2024
    def isRoomFor(self, which_clade: int) -> bool:
        if len(self._symbionts) == 0:                   return True
        if len(self._symbionts) == self._max_occupants: return False
        # make sure all present are of the same clade
        clade_check = self._symbionts[0].getCladeNumber()
        assert(all(clade_check == s.getCladeNumber() for s in self._symbionts[1:]))
        # now check that the requesting clade matches
        return which_clade == clade_check

    ######################################################
    # bgl: Feb|Mar 2024
    #def removeSymbiont(self, current_time: float) -> None:
    def removeSymbiont(self, symbiont: Symbiont) -> None:
        ''' method to remove the currently occuping symbiont from this host cell
        Parameters:
            symbiont: the symbiont to remove from this cell
        '''
        # bgl: Feb|Mar 2024
        assert(symbiont in self._symbionts)
        self._symbionts.remove(symbiont)
        self.updateNextEventTime()  # min next event time among remaining symbionts

        #self._symbiont = None
        #self._occupied = False

        # add to the residence time for this cell
        # bgl: Feb|Mar 2024
        #assert(self._last_occupied_time != INFINITY)
        #self._sum_residence_time += (current_time - self._last_occupied_time)
        #self._num_occupants += 1
        #self._last_occupied_time = INFINITY

    #########################################################################
    # bgl: Feb|Mar 2024
    #def setSymbiont(self, symbiont: 'Symbiont', current_time: float) -> None:
    def addSymbiont(self, symbiont: Symbiont) -> None:
        ''' updates this cell to have a new occupying symbiont
        Parameters:
            symbiont: a Symbiont object
        '''
        # it could be the case that we are swapping in a child symbiont
        # and evicting a parent, without ever calling removeSymbiont();
        # if that is the case, make sure to add in the unit-time rectangle
        # corresponding to the parent's time in the cell...
        # bgl: Feb|Mar 2024
        #if self._symbiont is not None:  # something was just evicted...
        #    self._sum_residence_time += (current_time - self._last_occupied_time)
        #    self._num_occupants += 1

        # bgl: Feb|Mar 2024
        #self._symbiont = symbiont
        #self._occupied = True
        #self._last_occupied_time = current_time   # new symbiont's residence starts now
        assert(len(self._symbionts) < self._max_occupants)
        self._symbionts.append(symbiont)
        self.updateNextEventTime()

    #################################
    def computeDemand(self) -> float:
        ''' compute photosynthetic demand expected by this host cell per unit time
        Returns:
            the photosynthetic demand required by this host cell (as a float)
        '''
        ## 12 Apr 2016
        # rather than fuzzing uniformly, use Normal with 95% of the data b/w 
        # (mu +/- mu*f) -- see implementation in rng.py
        m = Parameters.HOST_CELL_DEMAND
        f = Parameters.HCD_FUZZ  # assume to be % of the mean
        demand = RNG.fuzz(m, f, Stream.HOST_CELL_DEMAND)

        return demand 

    #########################
    def __str__(self) -> str:
        ''' str version of this Cell object, with row, col, demand, and symbionts
        Returns:
            this Cell object as a str
        '''
        # bgl: Feb|Mar 2024
        #symbiont_id = None if self._symbiont is None else self._symbiont.getID()
        #return f"({self._level},{self._row},{self._col}): demand: {round(self._demand, 3)}" + \
        #       f"\tsymbiont: {symbiont_id}"
        symbiont_ids = ','.join(str(s._id) for s in self._symbionts)
        return f"({self._level},{self._row},{self._col}): demand: {round(self._demand, 3)}" + \
               f"\tsymbionts: {symbiont_ids}"

    #########################
    def __repr__(self) -> str:
        ''' str version of this Cell object, with row, col, demand, and symbiont,
            just to allow for easy printing of a list of Cell objects
        Returns:
            this Cell object as a str
        '''
        return self.__str__()

################################################################################
class Sponge:
    ''' This class implements the 3D sponge environment for a collection of host 
        cells.  Essentially, this class is nothing more than a wrapper for a 3D
        list of Cell references.
    '''

    # bgl: Feb|Mar 2024
    __slots__ = ('_num_levels', '_num_rows', '_num_cols', '_cells', '_grid', '_max_occupancy')
    #__slots__ = ('_num_levels', '_num_rows', '_num_cols', '_cells', '_grid')

    def __init__(self, num_levels:   int, 
                       num_rows:     int, 
                       num_cols:     int, 
                       grid:         GridType,
                       # bgl: Feb|Mar 2024
                       max_per_cell: int = 5) -> None:
        ''' initializer for a Sponge object
        Parameters:
            num_levels: integer number of levels
            num_rows: integer number of rows in the 3D matrix of cells
            num_cols: integer number of columns
            grid: grid type -- GridType.SQUARE or GridType.HEX
        '''
        self._num_levels    : int      = num_levels
        self._num_rows      : int      = num_rows
        self._num_cols      : int      = num_cols
        self._grid          : GridType = grid
        # bgl: Feb|Mar 2024
        self._max_occupancy : int      = 0

        # assign the 3D list of Cell references 
        # bgl: Feb|Mar 2024
        #self._cells = [[[Cell(l,r,c) for c in range(num_cols)] \
        def _generate(l:int, r:int, c:int, m:int) -> Cell:
            self._max_occupancy += m
            return Cell(l, r, c, m)
        self._cells = [[[_generate(l, r, c, max_per_cell) \
                            for c in range(num_cols)] \
                            for r in range(num_rows)] \
                            for l in range(num_levels)]

    # bgl: Feb|Mar 2024
    def getMaxOccupancy(self) -> int:  return self._max_occupancy

    # bgl: Feb|Mar 2024
    #def getDimensions(self) -> tuple[int, int]:
    def getDimensions(self) -> tuple[int, int, int]:
        ''' returns the sponge dimensions
        Returns:
            a tuple containing the integer number of rows and columns
        '''
        return (self._num_levels, self._num_rows, self._num_cols)

    def getCell(self, level: int, row: int, col: int) -> Cell:
        ''' returns the Cell object at the given level, row, column
        Parameters:
            level: integer valued level of desired cell, in [0, num_levels - 1]
            row:   integer valued row of desired cell,   in [0, num_rows - 1]
            col:   integer valued columnof desired cell, in [0, num_cols - 1]
        Returns:
            Cell object @ (level,row,col)
        Raises:
            ValueError if any of the level, row, or col values are out of bounds
        '''
        if row < 0 or row >= self._num_rows or \
           col < 0 or col >= self._num_cols or \
           level < 0 or level >= self._num_levels:
            raise ValueError(f"Error in Sponge.getCell: ({level},{row},{col}) out of bounds")
        cell = self._cells[level][row][col]
        return cell

    # bgl: Feb|Mar 2024
    #def checkForOpenAdjacentCell(self, cell: Cell) -> Cell or None or SymbiontState:
    def checkForOpenAdjacentCell(self, cell: Cell, which_clade: int ) -> Cell or None or SymbiontState:
        ''' method to check for an open adjacent cell on successful mitosis
        Parameters:
            cell: Cell object from which to emanate the search for another cell
            which_clade: clade of the symbiont that will be occupying 
                (used for enforcing only same clade in any one cell)
        Returns:
            a Cell object that can be used for hosting the result of mitosis,
            or None if no open cell, or SymbiontState.CELL_OUTSIDE_ENVIRONMENT
            if the (modeled) open cell is determined to be outside the scope
            of our 2D grid of host cells (e.g., above the top row or below the
            bottom row)
        '''

        # bgl: Feb|Mar 2024
        # if there is room in the current cell, use it by default;  the new cell
        # from mitosis will (obviously) be of the same clade, so no need to 
        # check for clade match
        if cell.getNumOccupants() < cell.getMaxOccupants(): return cell

        # otherwise, not using current cell so need to find an adjacent cell
        level, row, col = cell.getLevelRowCol()

        if self._grid == GridType.SQUARE:
            if self._num_levels == 1: max_neighbors = 8
            else:                     max_neighbors = 26
            # _getNeighborCells* does not check for occupancy
            neighbor_cells = self._getNeighborCells_square(cell)

        elif self._grid == GridType.HEX:
            if self._num_levels == 1: max_neighbors = 6
            else:                     max_neighbors = 20
            # _getNeighborCells* does not check for occupancy
            neighbor_cells = self._getNeighborCells_hex(cell)

        RNG.shuffle(neighbor_cells, Stream.CHECK_FOR_OPEN_CELL)
        number_of_neighbor_cells = len(neighbor_cells)
        
        open_cell = None
        found     = False
        while not found and len(neighbor_cells) > 0:
            # grab cell from random shuffle (NB: .pop returns last in list)
            candidate_cell = neighbor_cells.pop()

            # bgl: Feb|Mar 2024
            #if not candidate_cell.isOccupied():
            if candidate_cell.isEmpty() or candidate_cell.isRoomFor(which_clade):
                open_cell = candidate_cell
                found = True
        
        # if there is an open cell and the parent lives on the boundary of 
        # the 2D square grid / 2D hexagonal grid, the occupied cell will be 
        # in the scope  of our model with probability 
        #       (# of neighbors) / max_neighbors
        #       - cell in a one-level   2D square grid has max  8 neighbors
        #       - cell in a one-level   2D hex    grid has max  6 neighbors
        #       - cell in a multi-level 2D square grid has max 26 neighbors
        #       - cell in a multi-level 2D hex    grid has max 20 neighbors

        ############
        # QUESTION: was the different approach to allow wrapping in 1-level case?
        #           if so, don't we need to do for hex as well as square?
        ############
        cell_at_boundary = number_of_neighbor_cells < max_neighbors
        #if open_cell is not None and (row == 0 or row == Parameters.NUM_ROWS - 1):  # this was logic for square, but only for 1 level

        ## CHANGE!  to allow option for new one in same cell
        #       -- if open cell is current, do nothing
        #       -- if open cell is not current, do logic below
        if open_cell is not cell:  # if open_cell is current cell, the child just goes there
            if open_cell is not None and cell_at_boundary: # this was logic for hex
                prob_inside_scope = number_of_neighbor_cells / max_neighbors
                prob_infects_outside = RNG.uniform(0, 1, Stream.INFECT_CELL_OUTSIDE) 
                if prob_infects_outside > prob_inside_scope:  
                    open_cell = SymbiontState.CELL_OUTSIDE_ENVIRONMENT
            
        return open_cell
    
    
    # bgl: Feb|Mar 2024
    #def getneighbors_square(self, cell: Cell) -> list:
    def _getNeighborCells_square(self, cell: Cell) -> list:
        ''' private method to find all neighbor cells of a given cell and return as list
        Returns: 
            a list of Cell objects that neighbor the given cell within a square grid
        '''
        level, row, col = cell.getLevelRowCol()
        neighbors = []

        positions = [(0,-1,-1),(0,-1,0),(0,-1,1), \
                     (0,0,-1),          (0,0,1), \
                     (0,1,-1),( 0,1,0),( 0,1,1)]
                
        positions_above = [(1,-1,-1),(1,-1,0),(1,-1,1), \
                            (1,0,-1), (1,0,0), (1,0,1), \
                            (1,1,-1), (1,1,0), (1,1,1)]
        
        positions_below = [(-1,-1,-1),(-1,-1,0),(-1,-1,1), \
                            (-1,0,-1), (-1,0,0), (-1,0,1), \
                            (-1,1,-1), (-1,1,0), (-1,1,1)]
        
        if self._num_levels == 1:
            pass
        elif level == 0 and self._num_levels > 1:
            positions = positions + positions_above
        elif level == self._num_levels-1:
            positions = positions + positions_below
        else:
            positions = positions + positions_above + positions_below

        for pos in positions:
            l = level + pos[0]; r = row + pos[1]; c = col + pos[2]
            if r < 0 or r >= self._num_rows or \
               c < 0 or c >= self._num_cols or \
               l < 0 or l >= self._num_levels:
                continue
            neighbors.append(self.getCell(l,r,c))
        
        return neighbors


    # bgl: Feb|Mar 2024
    #def getneighbors_hex(self, cell: Cell) -> list:
    def _getNeighborCells_hex(self, cell: Cell) -> list:
        ''' method to find all neighbor cells of a given cell and return as list
        Returns: 
            a list of Cell objects that neighbor the given cell within a hex grid
        '''
        level, row, col = cell.getLevelRowCol()

        even_row_off = [ (-1,-1), (-1, 0), \
                        ( 0,-1), ( 0, 1), \
                        ( 1,-1), ( 1, 0) ]
        odd_row_off  = [ (-1, 0), (-1, 1), \
                        ( 0,-1), ( 0, 1), \
                        ( 1, 0), ( 1, 1) ]
        
        neighbors = []
        row_offset = even_row_off if row % 2 == 0 else odd_row_off
        
        for off in row_offset:
            l = level; r = row + off[0]; c = col + off[1]
            if r < 0 or r >= self._num_rows or c < 0 or c >= self._num_cols:
                continue
            neighbors.append(self.getCell(l,r,c))

        if self._num_levels > 1:

            if level >= 0 and level < self._num_levels-1:
                for off in row_offset:
                    l = level + 1; r = row + off[0]; c = col + off[1]
                    if r < 0 or r >= self._num_rows or c < 0 or c >= self._num_cols:
                        continue
                    neighbors.append(self.getCell(l,r,c))
                neighbors.append(self.getCell(l,row,col)) # adding cell directly above given cell

            if level > 0 and level < self._num_levels:
                for off in row_offset:
                    l = level - 1; r = row + off[0]; c = col + off[1]
                    if r < 0 or r >= self._num_rows or c < 0 or c >= self._num_cols:
                        continue
                    neighbors.append(self.getCell(l,r,c))
                neighbors.append(self.getCell(l,row,col)) # adding cell directly below given cell
        
        return neighbors

##########################
if __name__ == "__main__":

    class Symbiont:
        def __init__(self, id_: int) -> None: self._id = id_
        def _getID(self) -> int: return self._id

    c = Cell(level = 2, row = 3, col = 4, max_occupants = 10)
    print(c)

    c.addSymbiont(Symbiont(33))
    c.addSymbiont(Symbiont(44))
    print(c)
       
    
