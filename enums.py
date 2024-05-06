from __future__ import annotations
from enum import Enum

################################################################################
class EventType(Enum):
    # The event types are in a particular order below for ordering events;
    # so, for example, if two events have the same time, an escape event
    # takes precedence over a digestion event, etc.
    EVENT_MIN_SENTINEL  = -1
    ESCAPE              =  0
    DIGESTION           =  1
    END_G0              =  2  # when division process (G1/S/G2/M) starts
    END_G1SG2M          =  3  # when division actually occurs, & G0 re-starts
    DENOUEMENT          =  4
    ARRIVAL             =  5
    # EVICTION = ...  # no separate eviction event: see symbiont.py comments
    # bgl: Feb|Mar 2024
    #EVENT_MAX_SENTINEL  =  6
    CANCELLED           =  6  # to allow for event cancellation
    EVENT_MAX_SENTINEL  =  7

    # bgl: Feb|Mar 2024
    def __lt__(self, other: EventType) -> bool: return self.value < other.value

    # if you define __eq__ without defining __hash__ the object becomes
    # unhashable and an error will be thrown if you try to hash it.
    #def __eq__(self, other: EventType) -> bool: return self.value == other.value


################################################################################
class Stream(Enum):
    ''' enumeration to identify different streams (one per stochastic component
        in the model) for the random number generator
    '''
    MITOTIC_CLASS              = 0
    END_G0                     = 1
    END_G1SG2M                 = 2
    TIME_G0_ESCAPE             = 3
    TIME_G1SG2M_ESCAPE         = 4
    TIME_DENOUEMENT            = 5
    CHECK_FOR_OPEN_CELL        = 6
    DIGESTION_VS_ESCAPE_G0     = 7
    DIGESTION_VS_ESCAPE_G1SG2M = 8
    EVICTION                   = 9
    HOST_CELL_DEMAND           = 10
    ARRIVALS                   = 11
    CLADE                      = 12
    ARRIVAL_AFFINITY           = 13
    OPEN_CELL_ON_ARRIVAL       = 14
    INFECT_CELL_OUTSIDE        = 15
    PHOTOSYNTHATE              = 16
    PHOTOPROD                  = 17
    MITOTIC_COST_RATE          = 18
    PHOTOSYNTHATE_MUTATION     = 19
    PHOTOPROD_MUTATION         = 20
    MITOTIC_COST_RATE_MUTATION = 21
    DIVISION_AFFINITY          = 22

################################################################################
class MutationType(Enum):
    ''' enumeration to identify mutation type '''
    DELETERIOUS = 0
    BENEFICIAL  = 1
    NO_MUTATION = 2

################################################################################
class Placement(Enum):
    ''' enumeration to characterize initial symbiont placement '''
    RANDOMIZE  = 0
    HORIZONTAL = 1
    VERTICAL   = 2
    QUADRANT   = 3

################################################################################
class GridType(Enum):
    SQUARE = 0
    HEX    = 1
    
################################################################################
class SymbiontState(Enum):
    # used in division cases on border
    CELL_OUTSIDE_ENVIRONMENT = -1
    # to identify Symbiont's arrival method
    ARRIVED_FROM_POOL        =  0
    ARRIVED_VIA_DIVISION     =  1
    # states used for computing surplus at event end
    IN_G0                    =  2
    IN_G1SG2M                =  3
    # states for what happens at end of G1SG2M
    CHILD_INFECTS_OUTSIDE    =  4
    CHILD_EVICTED            =  5
    CHILD_NO_AFFINITY        =  6  # 18 Mar 2017
    PARENT_INFECTS_OUTSIDE   =  7
    PARENT_EVICTED           =  8
    PARENT_NO_AFFINITY       =  9  # 18 Mar 2017
    BOTH_STAY                = 10
    # additional states to identify exit status
    STILL_IN_RESIDENCE       = 11
    DIGESTION_IN_G0          = 12
    DIGESTION_IN_G1SG2M      = 13
    ESCAPE_IN_G0             = 14
    ESCAPE_IN_G1SG2M         = 15
    DENOUEMENT_IN_G0         = 16
    DENOUEMENT_IN_G1SG2M     = 17
