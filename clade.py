###############################################################################
# This class implements the symbiotic "clade". Symbionts of a particular clade 
# have shared characteristics that differentiate them from other clades. 
# ----------------------------------------------------------------------------
# A clade has the following instance variables:
#       
#        _clade_number                    : identifying clade number
#
#        >> FUZZ VALUES <<
#        _residence_fuzz                 : fuzz factor for determining avg residence time
#        _g0_fuzz                        : fuzz factor for determining G0 length
#        _g1sg2m_fuzz                    : fuzz factor for determining G1SG2M length
#        _mcr_fuzz                       : fuzz factor for determining mcr
#        _ppr_fuzz                       : fuzz factor for determining ppr
#
#        >> DIVISION AND PHOTOSYNTHETIC PRODUCTION DEFAULTS << 
#        _mitotic_cost_rate              : default mcr for this clade
#        _photosynthetic_production_rate : default ppr for this clade
#        _photosynthetic_reduction       : default reduction value k used to reduce ppr moving lower in grid
#
#        _arrival_affinity_prob          : probability of phagocytosis on arrival from the pool
#        _division_affinity_prob         : probability of phagocytosis on division (mitosis)
#        
#        _avg_residence_time             : average time symbionts of this clade remain in host
#        _avg_g0_length                  : average length of G0
#        _avg_g1sg2m_length              : average length of G1SG2M
#        _g0_escape_prob                 : probability of symbiont escape during G0
#        _g1sg2m_escape_prob             : probability of symbiont escape during G1SG2M
#        _parent_eviction_prob           : probability of parent eviction on successful mitosis
#
#        _initial_surplus_shape          : shape for gamma distro for initial amount of photosynthate surplus
#        _initial_surplus_scale          : scale for gamma distro for initial amount of photosynthate surplus
#        _max_initial_surplus            : maximum photosynthate surplus upon entry
#        
#        >> MUTATION DEFAULT VARIABLES <<
#        _phenotypic_mutation_prob       : probability of mutation occurring
#        _deleterious_prob               : probability of an occurring mutation being deleterious
#        _beneficial_shape               : gamma shape for beneficial mutation
#        _beneficial_scale               : gamma scale for beneficial mutation
#        _deleterious_shape              : gamma shape for deleterious mutation
#        _deleterious_scale              : gamma scale for deleterious mutation
#
# The Clade class also has the following class-level variable:
#        clade_objects                   : a list containing all clade objects, set at runtime
###############################################################################

import random
import logging
import copy

#from parameters import *

class Clade:
    ''' class to implement/store clade-level specific values, i.e., values that
        all symbionts of a particular clade will have
    '''

    ################################################
    __slots__ = (
        '_clade_number', \
        '_residence_fuzz', \
        '_g0_fuzz', \
        '_g1sg2m_fuzz', \
        '_ppr_fuzz', \
        '_mcr_fuzz', \
        '_avg_residence_time', \
        '_g0_length', \
        '_g1sg2m_length', \
        '_g0_escape_prob', \
        '_g1sg2m_escape_prob', \
        '_parent_eviction_prob', 
        '_photosynthetic_production_rate', \
        '_mitotic_cost_rate', \
        '_photosynthetic_reduction', \
        '_arrival_affinity_prob', \
        '_division_affinity_prob', \
        '_initial_surplus_shape', \
        '_initial_surplus_scale', \
        '_max_initial_surplus', \
        '_phenotypic_mutation_prob', \
        '_deleterious_prob', \
        '_beneficial_shape', \
        '_beneficial_scale', \
        '_deleterious_shape', \
        '_deleterious_scale', \
    )

    # class-level list of Clade objects 
    clade_objects = []

    ################################################
    @classmethod
    def getClade(cls, clade_number: int) -> 'Clade':
        ''' class-level method to return a particular clade object from the
            array number should be an integer between 0 and (numClades-1)
        Parameters:
            clade_number: integer valued number of clade
        Returns:
            a Clade object
        '''
        return cls.clade_objects[clade_number]

    ################################################
    @classmethod
    def addClade(cls, clade: 'Clade') -> None:
        ''' class-level method to append a Clade object to the Clade class
            internal list of clades
        Parameters:
            clade: a Clade object
        '''
        assert(isinstance(clade, Clade))
        cls.clade_objects.append(clade)

    ##############################################
    def __init__(self, clade_number: int) -> None:
        ''' initializer for a Clade object
        Parameters:
            clade_number: index (sequential, from 1) of this clade
        '''
        self._clade_number : int = clade_number

        # start with default None values for the clade, which will then be
        # overwritten via setters/mutators (see below) when reading from 
        # CSV input file
        self._residence_fuzz                : float = None
        self._g0_fuzz                       : float = None
        self._g1sg2m_fuzz                   : float = None
        self._ppr_fuzz                      : float = None  
        self._mcr_fuzz                      : float = None 

        self._avg_residence_time            : float = None
        self._g0_length                     : float = None
        self._g1sg2m_length                 : float = None
        self._g0_escape_prob                : float = None
        self._g1sg2m_escape_prob            : float = None
        self._parent_eviction_prob          : float = None

        self._photosynthetic_production_rate: float = None
        self._mitotic_cost_rate             : float = None

        # k=2: production rate in bottom row will be _photo_prod_rate[i] / k
        # with linear decrease from _photo_prod_rate[i] (top row) to 
        # _photo_prod_rate[i] / k (bottom row)
        self._photosynthetic_reduction      : float = None

        self._arrival_affinity_prob         : float = None
        self._division_affinity_prob        : float = None

        # different gamma values -- control mean via shape
        #      mean = shape*scale
        #      sd   = sqrt(shape*scale^2)
        self._initial_surplus_shape   : float = None   # for gamma
        self._initial_surplus_scale   : float = None   # for gamma
        self._max_initial_surplus     : float = None

        # probability of phenotypic mutation on division (10^-6 or 10^-7 more likely in practice)
        self._phenotypic_mutation_prob: float = None
        self._deleterious_prob        : float = None   # prob of mutation being deleterious
        self._beneficial_shape        : float = None   # shape for beneficial gamma (see comment in rng.py:divfuzz)
        self._beneficial_scale        : float = None   # scale for beneficial gamma (")
        self._deleterious_shape       : float = None   # shape for deleterious gamma
        self._deleterious_scale       : float = None   # scale for deleterious gamma

    ############################################################################
    ''' simple setter/mutator methods '''
    def setCladeNumber(self, value: int)                    -> None: self._clade_number                   = value
    def setResidenceFuzz(self, value: float)                -> None: self._residence_fuzz                 = value
    def setG0Fuzz(self, value: float)                       -> None: self._g0_fuzz                        = value
    def setG1SG2MFuzz(self, value: float)                   -> None: self._g1sg2m_fuzz                    = value
    def setPhotosyntheticProductionRate(self, value: float) -> None: self._photosynthetic_production_rate = value
    def setMitoticCostRate(self, value: float)              -> None: self._mitotic_cost_rate              = value
    def setPPRFuzz(self, value: float)                      -> None: self._ppr_fuzz                       = value
    def setMCRFuzz(self, value: float)                      -> None: self._mcr_fuzz                       = value
    def setArrivalAffinityProb(self, value: float)          -> None: self._arrival_affinity_prob          = value
    def setDivisionAffinityProb(self, value: float)         -> None: self._division_affinity_prob         = value
    def setAvgResidenceTime(self, value: float)             -> None: self._avg_residence_time             = value
    def setG0Length(self, value: float)                     -> None: self._g0_length                      = value
    def setG1SG2MLength(self, value: float)                 -> None: self._g1sg2m_length                  = value
    def setG0EscapeProb(self, value: float)                 -> None: self._g0_escape_prob                 = value
    def setG1SG2MEscapeProb(self, value: float)             -> None: self._g1sg2m_escape_prob             = value
    def setParentEvictionProb(self, value: float)           -> None: self._parent_eviction_prob           = value
    def setPhotosyntheticReduction(self, value: float)      -> None: self._photosynthetic_reduction       = value
    def setInitialSurplusShape(self, value: float)          -> None: self._initial_surplus_shape          = value
    def setInitialSurplusScale(self, value: float)          -> None: self._initial_surplus_scale          = value
    def setMaxInitialSurplus(self, value: float)            -> None: self._max_initial_surplus            = value
    def setPhenotypicMutationProb(self, value: float)       -> None: self._phenotypic_mutation_prob       = value
    def setDeleteriousProb(self, value: float)              -> None: self._deleterious_prob               = value
    def setBeneficialShape(self, value: float)              -> None: self._beneficial_shape               = value
    def setBeneficialScale(self, value: float)              -> None: self._beneficial_scale               = value
    def setDeleteriousShape(self, value: float)             -> None: self._deleterious_shape              = value
    def setDeleteriousScale(self, value: float)             -> None: self._deleterious_scale              = value

    ############################################################################
    ''' simple getter/accessor methods '''
    def getCladeNumber(self)             -> int:     return self._clade_number
    def getResidenceFuzz(self)           -> float:   return self._residence_fuzz
    def getG0Fuzz(self)                  -> float:   return self._g0_fuzz
    def getG1SG2MFuzz(self)              -> float:   return self._g1sg2m_fuzz
    def getPPR(self)                     -> float:   return self._photosynthetic_production_rate
    def getMCR(self)                     -> float:   return self._mitotic_cost_rate
    def getPPRFuzz(self)                 -> float:   return self._ppr_fuzz
    def getMCRFuzz(self)                 -> float:   return self._mcr_fuzz
    def getArrivalAffinityProb(self)     -> float:   return self._arrival_affinity_prob
    def getDivisionAffinityProb(self)    -> float:   return self._division_affinity_prob
    def getAvgResidenceTime(self)        -> float:   return self._avg_residence_time
    def getG0Length(self)                -> float:   return self._g0_length
    def getG1SG2MLength(self)            -> float:   return self._g1sg2m_length
    def getG0EscapeProb(self)            -> float:   return self._g0_escape_prob
    def getG1SG2MEscapeProb(self)        -> float:   return self._g1sg2m_escape_prob
    def getParentEvictionProb(self)      -> float:   return self._parent_eviction_prob
    def getPhotosyntheticReduction(self) -> float:   return self._photosynthetic_reduction
    def getInitialSurplusShape(self)     -> float:   return self._initial_surplus_shape
    def getInitialSurplusScale(self)     -> float:   return self._initial_surplus_scale
    def getMaxInitialSurplus(self)       -> float:   return self._max_initial_surplus
    def getPhenotypicMutationProb(self)  -> float:   return self._phenotypic_mutation_prob
    def getDeleteriousProb(self)         -> float:   return self._deleterious_prob
    def getBeneficialShape(self)         -> float:   return self._beneficial_shape
    def getBeneficialScale(self)         -> float:   return self._beneficial_scale
    def getDeleteriousShape(self)        -> float:   return self._deleterious_shape
    def getDeleteriousScale(self)        -> float:   return self._deleterious_scale

    ############################################################################
    def __str__(self) -> str:
        ''' returns a str version of this Clade object '''
        string  = f"CLADE {self._clade_number} PARAMETER VALUES:\n"
        #string += f"\t_clade_number                   : {self._clade_number}\n"
        string += f"\t_photosynthetic_production_rate : {self._photosynthetic_production_rate}\n"
        string += f"\t_mitotic_cost_rate              : {self._mitotic_cost_rate}\n"
        string += f"\t_photosynthetic_reduction       : {self._photosynthetic_reduction}\n"
        string += f"\t_arrival_affinity_prob          : {self._arrival_affinity_prob}\n"
        string += f"\t_division_affinity_prob         : {self._division_affinity_prob}\n"
        string += f"\t_residence_fuzz                 : {self._residence_fuzz}\n"
        string += f"\t_g0_fuzz                        : {self._g0_fuzz}\n"
        string += f"\t_g1sg2m_fuzz                    : {self._g1sg2m_fuzz}\n"
        string += f"\t_ppr_fuzz                       : {self._ppr_fuzz}\n"
        string += f"\t_mcr_fuzz                       : {self._mcr_fuzz}\n"
        string += f"\t_avg_residence_time             : {self._avg_residence_time}\n"
        string += f"\t_g0_length                      : {self._g0_length}\n"
        string += f"\t_g1sg2m_length                  : {self._g1sg2m_length}\n"
        string += f"\t_g0_escape_prob                 : {self._g0_escape_prob}\n"
        string += f"\t_g1sg2m_escape_prob             : {self._g1sg2m_escape_prob}\n"
        string += f"\t_parent_eviction_prob           : {self._parent_eviction_prob}\n"
        string += f"\t_initial_surplus_shape          : {self._initial_surplus_shape}\n"
        string += f"\t_initial_surplus_scale          : {self._initial_surplus_scale}\n"
        string += f"\t_max_initial_surplus            : {self._max_initial_surplus}\n"
        string += f"\t_phenotypic_mutation_prob       : {self._phenotypic_mutation_prob}\n"
        string += f"\t_deleterious_prob               : {self._deleterious_prob}\n"
        string += f"\t_beneficial_shape               : {self._beneficial_shape}\n"
        string += f"\t_beneficial_scale               : {self._beneficial_scale}\n"
        string += f"\t_deleterious_shape              : {self._deleterious_shape}\n"
        string += f"\t_deleterious_scale              : {self._deleterious_scale}"
        return string
