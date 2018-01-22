'''Base class for simulation messages.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

import six, abc

@six.add_metaclass(abc.ABCMeta)
class SimulationMessage(object):
    '''An abstract base class for simulation messages

    Each simulation event contains a simulation message. All simulation messages are objects. This
    class supports compact declaration of `SimulationMessage` types. For example::

        GivePopulation = SimulationMsgUtils.create('GivePopulation',
            """A WC simulator message sent by a species pop object ... """, ['population'])

    defines a `GivePopulation` message (with an elided docstring).

    Attributes:
        __slots__: list: use `__slots__` to save memory because a simulation may contain many messages
    '''
    __slots__ = []

    def __init__(self, *args):
        '''Initialize a `SimulationMessage`

        `SimulationMessage` subclasses are defined by `SimulationMsgUtils.create()`.

        Args:
            args: tuple: argument list for initializing a subclass instance
        '''
        if len(args) != len(self.__slots__):
            raise ValueError("Constructor for SimulationMessage '{}' expects {} argument(s), but "
                "{} provided".format(
                self.__class__.__name__, len(self.__slots__), len(args)))
        for slot,arg in zip(self.__slots__, args):
            setattr(self, slot, arg)

    def __str__(self):
        '''Provide a string representation of a `SimulationMessage`.
        '''
        vals = []
        for attr in self.__slots__:
            if hasattr(self, attr):
                vals.append(getattr(self,attr))
            else:
                vals.append('undef')
        # todo: improve
        # we use str(dict()) to distinguish numeric and string attrs, but otherwise this stinks
        values = {attr:val for attr,val in zip(self.__slots__,vals)}
        return "SimulationMessage: {}({})".format(self.__class__.__name__, values)

class SimulationMsgUtils(object):
    '''Utilities for creating `SimulationMessage` types
    '''

    @staticmethod
    def create(name, docstring, attributes=None):
        '''Define a subclass of `SimulationMessage`

        Args:
            name: string: the name of the class being defined
            docstring: string: a docstring for the class
            attributes: optional list: the attributes for the subclass which is being defined

        Raises:
            ValueError: if `docstring` is empty
        '''
        if docstring == '':
            raise ValueError("SimulationMessage docstring cannot be empty")
        attrs = {}
        if attributes is not None:
            attrs['__slots__'] = attributes
        generated_simulation_message_cls = type(name, (SimulationMessage,), attrs)
        generated_simulation_message_cls.__doc__ = docstring

        '''
        todo: would like to define the class as an attribute of the caller's module so the
        caller doesn't have to. unfortunately inspect doesn't support this. perhaps possible if
        messages are not declared in __main__ module
        '''
        return generated_simulation_message_cls
