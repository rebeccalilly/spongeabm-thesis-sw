from __future__ import annotations

from heapq import heappush, heappop
from enums import EventType

################################################################################
class Event:
    ''' Class to implement an event (to be stored in the corresponding event list)
        in the simulation model.  An event is determined by its time, the type
        of event, and the corresponding symbiont driving the event.
    '''
    __slots__ = ('_time', '_type', '_symbiont', '_event_num')

    # class-level variable to track the number of total eventds
    _event_cnt : int = 0

    #####################################
    def __init__(self, time: float, event_type: EventType, symbiont: Symbiont) -> None:
        ''' initialize for a simulation event
        Parameters:
            time: time the event is to occur
            event_type: the type of event (see EventType class)
            symbiont: the symbiont driving the event
        '''
        self._time     : float      = time
        self._type     : EventType  = event_type
        self._symbiont : Symbiont   = symbiont
        self._event_num: int        = Event._event_cnt

        Event._event_cnt += 1

    #####################################
    def __lt__(self, other: Event) -> bool:
        ''' method to compare this event to another
        Parameters:
            other: an Event object to compare to
        Returns:
            True if this event should appear before the other event, False o/w
        '''
        # sort first on event time, then on event type, then on event number (JIC)
        return (self._time,self._type,self._event_num) \
            < (other._time,other._type,other._event_num)

    #####################################
    ''' simple getter/accessor methods '''
    def getType(self)     -> EventType:  return self._type
    def getTime(self)     -> float:      return self._time 
    def getSymbiont(self) -> Symbiont:   return self._symbiont

    ###########################
    def __str__(self) -> str:
        ''' returns an str representation of this Event object '''
        #return f"{self._type.name} @ t= {self._time} :\n\t{self._symbiont}"
        return f"(s{self._symbiont._id}):{self._type.name} @ t={self._time:.3f}"

    ###########################
    # bgl: Feb|Mar 2024
    def __repr__(self) -> str:
        ''' returns an str representation of this Event object '''
        return str(self)


###############################################################################
class EventList:
    ''' Class to implement an event list for the simulation model, using a
        Python list initially, but converted to a priority queue using
        heapq.heappush and heapq.heappop.  This facilitates efficient insertion
        and removal of time-sequenced events as part of the event calendar 
        (event list).
    '''
    # bgl: Feb|Mar 2024
    #__slots__ = ('_heap')
    __slots__ = ('_heap', '_event_finder')

    ###########################
    def __init__(self) -> None:
        ''' initializer, creating an empty event list (for heap) 
            and a dictionary to help easily access (and cancel, rather than
            remove) events 

            https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
        '''
        self._heap = []
        self._event_finder = {}  # bgl: Feb|Mar 2024

    ##########################################
    def getNextEvent(self) -> Event | None:
        ''' returns the next event to occur in simulated time
        Returns:
            an Event object corresponding to the next event to occur
        Notes:
            any cancelled events on the calendar are simply popped
                and ignored
        '''
        event = None

        # bgl: Feb|Mar 2024
        #if len(self._heap) > 0:
        #    event = heappop(self._heap)

        # bgl: Feb|Mar 2024
        if len(self._heap) > 0:
            # loop until a non-cancelled event is popped
            done = False
            while not done:
                event = heappop(self._heap)
                if event._type != EventType.CANCELLED:
                    key = (event._time, event._type, event._symbiont.getID())
                    self._event_finder.pop(key)  # remove from the finder dict
                    done = True

        return event   # empty list returns None

    ##############################################
    def insertEvent(self, event: Event) -> None:
        ''' inserts a new event in order of event time into the event list
        Parameters:
            event: an Event object (w/ info time, event type, associated symbiont)
        '''
        assert(event != None)
        heappush(self._heap, event)

        # bgl: Feb|Mar 2024
        # add the event to the dictionary to make cancelling easy (if req'd)
        key = (event._time, event._type, event._symbiont.getID())
        self._event_finder[key] = event


    ##############################################
    # bgl: Feb|Mar 2024
    def cancelEvent(self, time: float, type_: EventType, symbiont_id: int) -> None:
        ''' marks an event as cancelled (rather than explicitly removing at
            this point, which would break the heap invariant); the cancelled
            event will be ignored by getNextEvent whenever it is popped from
            the heap in priority order
        Parameters:
            time:        the time of the event to cancel
            type:        the type of the event to cancel
            symbiont_id: integer ID of the symbiont associated with the event
        '''
        key = (time, type_, symbiont_id)
        # grab the event reference using the dict finder
        event = self._event_finder[key]
        # now change its type to CANCELLED, which will just be ignored
        # by getNextEvent
        event._type = EventType.CANCELLED

        # also just dump the entry in the finder, since we no longer need it
        self._event_finder.pop(key)

    ##############################################
    # bgl: Feb|Mar 2024
    def changeEventTime(self, old_time   : float, 
                              new_time   : float,
                              type_      : EventType,
                              symbiont_id: int) -> None:
        ''' method to change the time of an already-scheduled event in the event
            list, e.g., to move a digestion/escape event earlier in simulated
            time
        Parameters:
            old_time:    the time of the event to change, as it currently appears
                                in the event list
            new_time:    the updated time of the event list
            type:        the type of the event to modify
            symbiont_id: integer ID of the symbiont associated with the
        '''
        key = (old_time, type_, symbiont_id)
        # grab the event reference using the dict finder
        event = self._event_finder[key]
        assert(event._time == old_time)
        event._time = new_time
    

    #########################
    def __len__(self) -> int:
        ''' the current length of the event list
        Returns:
            the integer valued number of events in the event list
        '''
        return len(self._heap)

    #########################
    # bgl: Feb|Mar 2024
    def __str__(self) -> str:
        events = []
        heap = self._heap.copy()
        while len(heap) > 0: events.append(heappop(heap))
        return str(events)
        

##########################
if __name__ == "__main__":
    # some insert / cancel / remove tests
    elist = EventList()

    from symbiont import Symbiont
    from sponge import Sponge
    from enums import GridType, EventType
    from parser import Parser
    from parameters import Parameters

    Parser.parseCSVInput("input.csv")

    sponge = Symbiont.sponge = \
        Sponge(num_levels = 3, num_rows = 5, num_cols = 5, \
               grid = GridType.SQUARE, max_per_cell = 10)
    cell = sponge.getCell(level = 2, row = 3, col = 4)

    # insert symbiont
    times = [3.5, 2.4, 1.0, 2.2, 1.9, 3.7, 2.9, 0.5, 1.3]
    symbs = []
    for t in times:
        s = Symbiont(0, cell, t)
        event = Event(t, EventType.END_G0, s)
        elist.insertEvent(event)
        symbs.append(s)
        print(elist)
        print('-' * 80)

    # cancel symbiont 2's event @ t=1.0, replacing with another
    offset = 0.0  # 0.05
    elist.cancelEvent(1.0, EventType.END_G0, 2)
    event = Event(1.0 + offset, EventType.DIGESTION, symbs[2])
    elist.insertEvent(event)
    print(elist)
    print('-' * 80)

    # cancel symbiont 4's event @ t=1.9, replacing with another
    elist.cancelEvent(1.9, EventType.END_G0, 4)
    event = Event(1.9 + offset, EventType.DIGESTION, symbs[4])
    elist.insertEvent(event)
    print(elist)
    print('-' * 80)

    # cancel symbiont 1's event @ t=2.4, replacing with another
    elist.cancelEvent(2.4, EventType.END_G0, 1)
    event = Event(2.4 + offset, EventType.DIGESTION, symbs[1])
    elist.insertEvent(event)
    print(elist)
    print('-' * 80)
    
    # blow through the list -- should ignore any cancelled
    event = elist.getNextEvent()
    while event is not None:
        print(event)
        event = elist.getNextEvent()

    print('-' * 80)
    print(elist)
    print(elist._event_finder)





