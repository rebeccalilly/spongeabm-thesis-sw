#####################################
class Parameters:
    ''' Class to allow global-to-the-simulation parameters to be acccessed
        directly from across the ABM.  These values will be updated by
        parser.py when parameter values are read from input CSV.
    '''

    global INFINITY
    INFINITY = float('inf')

    INITIAL_SEED:              int         = 0
    MAX_SIMULATED_TIME:        float       = 0.0    # in days 
    NUM_LEVELS:                int         = 0
    NUM_ROWS:                  int         = 0
    NUM_COLS:                  int         = 0 
    GRID_TYPE:                 'GridType'  = None   # 'square' or 'hex'
    NUM_INITIAL_SYMBIONTS:     int         = 0
    INITIAL_PLACEMENT:         'Placement' = None   # 'randomize', 'vertical', or 'horizontal'
    HOST_CELL_DEMAND:          float       = 0.0
    HCD_FUZZ:                  float       = 0.0
    AVG_TIME_BETWEEN_ARRIVALS: float       = 0.0

    NUM_CLADES:                int         = 0
    CLADE_PROPORTIONS:         list[float] = []

    POPULATION_FILENAME:       str         = ""
    WRITE_CSV_INFO:            bool        = False
    CSV_FILENAME:              str         = ""
    WRITE_LOGGING_INFO:        bool        = False
    LOG_FILENAME:              str         = ""

    PRINT_PARAMETER_VALUES:    bool        = False

    @classmethod
    def printParameters(cls) -> None:
        ''' class-level method to print out values of simulation-level
            parameters '''
        for var in dir(cls):
            if not var.startswith('__') and not callable(getattr(cls, var)):
                value = eval(f"cls.{var}")
                print(f"{var:<30}: {value}")
