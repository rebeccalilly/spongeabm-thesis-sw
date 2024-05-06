import numpy.typing
from numpy.random import MT19937, Generator
from parameters import *
from enums import Stream, MutationType

######################################################################
class RNG:
    ''' This class implements a wrapper around numpy's MT19937 generator
        to allow for a "streams" implementation, i.e., where we can have a
        different stream of random numbers for each different stochastic
        component.  The stream will be indicated using one of the values
        defined in the Stream enumeration class.  Each wrapper method will do
        the right thing to pull and then update the state of the particular
        stream.
    '''

    # class-level variables
    _streams: list[numpy.random.Generator] = []  # not yet initialized
    _initialized: bool = False

    ############################################################################
    @classmethod
    def initializeStreams(cls) -> None:
        ''' Class-level method to initialize streams for generating random
            numbers.  This uses the .jumped() method to set up the streams
            sufficiently far apart, giving us one stream per stochastic
            component (i.e., number of entries in the Stream enum).

            See:
                https://bit.ly/numpy_random_jumping
                https://bit.ly/numpy_random_Generator
        '''
        cls._streams = []
        rng = MT19937(Parameters.INITIAL_SEED)  # Mersenne twister
        for i in range(len(Stream)):
            cls._streams.append(Generator(rng.jumped(i)))
        cls._initialized = True

    ############################################################################
    @classmethod
    def randint(cls, a: int, b: int, which_stream: Stream) -> numpy.int64:
        ''' class-level method to generate integers uniformly between a and b
            inclusive
        Parameters:
            a: minimum-value integer in the range
            b: maximum-value integer in the range
            which_stream: named entry from Stream class
        Returns:
            a uniformly generated integer in [a,b]
        '''
        if not isinstance(which_stream, Stream):
            raise TypeError(f"in RNG.randint, which_stream must be of type Stream, not {type(which_stream)}")
        if not cls._initialized: cls.initializeStreams()
        return cls._streams[which_stream.value].integers(a, b, endpoint = True)
        #                                   b inclusive: ^^^^^^^^^^^^^^^ 

    ############################################################################
    @classmethod
    def random(cls, which_stream: Stream, exclude_zero: bool = False) -> numpy.float64:
        ''' class-level method to generate floating-point values uniformly
            in [0,1), or in (0,1) if exclude_zero is True
        Parameters:
            which_stream: named entry from Stream class
            exclude_zero: if True, allows numpy's random() to return 0 (default)
        Returns:
            a uniformly generated floating point value in either [0,1) or (0,1)
        '''
        if not isinstance(which_stream, Stream):
            raise TypeError(f"in RNG.random, which_stream must be of type Stream, not {type(which_stream)}")
        if not cls._initialized: cls.initializeStreams()
        value = cls._streams[which_stream.value].random()
        if exclude_zero:
            while value == 0:  
                value = cls._streams[which_stream.value].random()
        return value
    
    ############################################################################
    @classmethod
    def uniform(cls, a: float, b: float, which_stream: Stream, exclude_a: bool = False) -> numpy.float64:
        ''' class-level method to generate floating-point values uniformly
            in [a,b), or in (a,b) if exclude_a is True
        Parameters:
            a: floating-point minimum value of the distribution
            b: floating-point maximum value of the distribution
            which_stream: named entry from Stream class
            exclude_a: if True, allows numpy's uniform(a,b) to return a (default)
        Returns:
            a uniformly generated floating point value in either [a,b) or (a,b)
        '''
        if not isinstance(which_stream, Stream):
            raise TypeError(f"in RNG.uniform, which_stream must be of type Stream, not {type(which_stream)}")
        if not cls._initialized: cls.initializeStreams()
        value = cls._streams[which_stream.value].uniform(a,b)
        if exclude_a:
            while value == a:
                value = cls._streams[which_stream.value].uniform(a,b)
        return value
    
    ############################################################################
    @classmethod
    def exponential(cls, mu: float, which_stream: Stream) -> numpy.float64:
        ''' Class-level method to generate variates drawn from an exponential
            distribution with given mean mu.
        Parameters:
            mu: float representing the mean (scale), not rate (e.g., avt time
                b/w arrivals rather arrivals per unit time)
            which_stream: named entry from Stream class
        Returns:
            a floating point value drawn from an exponential(mu) distribution
        '''
        if not isinstance(which_stream, Stream):
            raise TypeError(f"in RNG.exponential, which_stream must be of type Stream, not {type(which_stream)}")
        if not cls._initialized: cls.initializeStreams()
        return cls._streams[which_stream.value].exponential(mu)  # expects mean, not rate

    ############################################################################
    @classmethod
    def gamma(cls, shape: float, scale: float, which_stream: Stream) -> numpy.float64:
        ''' Class-level method to generate variates drawn from a gamma
            distribution with given shape and scale.
        Parameters:
            shape: float value for the gamma shape parameter
            scale: float value for the gamma scale parameter (default 1.0)
            which_stream: named entry from Stream class
        Returns:
            a floating point value drawn from a gamma(shape,scale) distribution
        '''
        if not isinstance(which_stream, Stream):
            raise TypeError(f"in RNG.gamma, which_stream must be of type Stream, not {type(which_stream)}")
        if not cls._initialized: cls.initializeStreams()
        return cls._streams[which_stream.value].gamma(shape, scale)
    
    ############################################################################
    @classmethod
    def normal(cls, mu: float, s: float, which_stream: Stream) -> numpy.float64:
        ''' Class-level method to generate variates drawn from a normal
            distribution with mean mu and standard deviation s.
        Parameters:
            mu: float value for the normal's mean parameter
            s: float value for the normal's standard deviation parameter
            which_stream: named entry from Stream class
        Returns:
            a floating point value drawn from a normal(mu,s) distribution
        '''
        if not isinstance(which_stream, Stream):
            raise TypeError(f"in RNG.normal, which_stream must be of type Stream, not {type(which_stream)}")
        if not cls._initialized: cls.initializeStreams()
        return cls._streams[which_stream.value].normal(mu, s)
    
    ############################################################################
    @classmethod
    def fuzz(cls, mean: float, fuzz_pct: float, which_stream: Stream) -> numpy.float64:
        ''' class-level method to fuzz a particular value given a mean and some
            fuzz pct. Uses normal for fuzzing: given the fuzz pct f relative
            to mean m:
                      m +/- mf = m +/- 2s   ==>   s = mf/2
            Then generate normal(m, s) -- just ensure we don't go negative!
            This will give ~95% of the values within m +/- (f%  of m).
        Parameters:
            mean: floating point value for the normal distribution's mean parameter
            fuzz_pct: floating point fuzz pct (see description above)
            which_stream: named entry from Stream class
        Returns:
            floating point value of appropriately fuzzed normal
        '''
        if not isinstance(which_stream, Stream):
            raise TypeError(f"in RNG.fuzz, which_stream must be of type Stream, not {type(which_stream)}")
        sd = (mean * fuzz_pct) / 2
        value = -1
        while value < 0:  # there are better ways to ensure not negative... :(
            value = cls.normal(mean, sd, which_stream)
        return value

    #############################################################################
    # NEW VERSION OF divfuzz AFTER SPRING 2016 MEETING
    @classmethod
    def divfuzz(cls, value: float, clade: 'Clade', which_stream: Stream) -> tuple[numpy.float64, MutationType]:
        ''' class-level method Method to fuzz a particular value on symbiont division

            Approach:
            - flip coin to determine whether 
                (p)   phenotypic mutation or
                (1-p) no phenotypic mutation or silent
            - if phenotypic mutation, flip another coin to determine
                (q)   deleterious [typically larger proportion] or 
                (1-q) beneficial
    
            For modeling beneficial mutations, use gamma distribution with
            distribution mean (mu^+) such that, for example, 75% are a 1.5%
            increase or less relative to the parent's value, and the max
            beneficial change can be 10% relative to parent's value.
    
            For modeling deleterious mutations, use gamma distribution with
            distribution mean (mu^-) such that, for example, 50% are a 2.0%
            decrease or less relative to the parent's value, and there is no
            max deleterious value (subject to physical constraints, e.g., can't
            have negative PPR, MCR, surplus).
    
            Need seven corresponding parameters in parameters.py: 
                p: PHENOTYPIC_MUTATION_PROB
                q: DELETERIOUS_PROB
                two %'s for beneficial, max % for beneficial
                two %'s for deleterious
    
            NOTE: a deleterious MCR adds; all other deleterious subtract.

        Parameters:
            value: floating point value to be fuzzed
            clade: an object of class Clade identifying the clade (for clade-
                   specific phenotypic mutation probabilities)
            which_stream: named entry from Stream class
        Returns:
            Feturns the floating-point fuzz amount (which may be zero in the
            case of no phenotypic mutation), which must be added or subtracted
            on the calling side (in symbiont.py).
        '''
        if not isinstance(which_stream, Stream):
            raise TypeError(f"in RNG.divfuzz, which_stream must be of type Stream, not {type(which_stream)}")
        if not cls._initialized: cls.initializeStreams()
    
        mutation = MutationType.NO_MUTATION # default
        fuzzamt  = 0

        phenotypic_mutation_prob = cls.random(which_stream)
        if phenotypic_mutation_prob < clade.getPhenotypicMutationProb():
            # then a mutation occurs... of those most will be deleterious
            deleterious_prob = cls.random(which_stream)
            if deleterious_prob < clade.getDeleteriousProb():
                # mutation will be a deleterious one: Gamma(2,1/0.83915) will give
                # 75% of value 1.5 or less -- use the generated value z as the
                # pct of the value v to change (e.g., v + v*(z/100));
                # Nasheya and Barry empirically determined shape=2, rate=0.83915
                # (scale = 1/0.83915) as appropriate parameters -- 23 Sep 2016
                # ensure no negative values (i.e., the multiplied % is < 100%)
                variate = 100
                while variate >= 100:
                    variate = cls.gamma(clade.getDeleteriousShape(), \
                                        clade.getDeleteriousScale(), \
                                        which_stream)
                fuzzamt = (value * variate/100.0)
                #print(">>>>>>>>> DEL= ",fuzzamt)
                mutation = MutationType.DELETERIOUS
            else:
                # mutation will be a beneficial one: Gamma(2,1/1.795) will give
                # 75% of value 1.5 or less -- use the generated value z as the
                # pct of the value v to change (e.g., v + v*(z/100));
                # Nasheya and Barry empirically determined shape=2, rate=1.795
                # (scale = 1/1.795) as appropriate parameters, and confirmed by
                # comparing Python's gammavariate with R's dgamma -- 23 Sep 2016;
                # recall that M&A said max beneficial change is 10%
                variate = 100
                while variate > 10:  # accept-reject... there are better ways :(
                    variate = cls.gamma(clade.getBeneficialShape(), \
                                        clade.getBeneficialScale(), \
                                        which_stream)
                    fuzzamt = (value * variate/100.0)
                #print(">>>>>>>>> BEN= ",fuzzamt)
                mutation = MutationType.BENEFICIAL

        return (fuzzamt, mutation)

    #############################################################################
    @classmethod
    def shuffle(cls, array: list, which_stream: Stream) -> None:
        ''' class-level method to shuffle a given list in place
        Parameters:
            array: a Python list
            which_stream: named entry from Stream class
        '''
        if not isinstance(which_stream, Stream):
            raise TypeError(f"in RNG.shuffle, which_stream must be of type Stream, not {type(which_stream)}")
        if not cls._initialized: cls.initializeStreams()
        cls._streams[which_stream.value].shuffle(array)


