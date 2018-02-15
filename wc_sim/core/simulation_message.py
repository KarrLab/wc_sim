""" Base class for simulation messages.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import six, abc


@six.add_metaclass(abc.ABCMeta)
class SimulationMessage(object):
    """ An abstract base class for simulation messages

    Each simulation event contains a simulation message. All simulation messages are objects. This
    module supports compact declaration of `SimulationMessage` types. For example::

        GivePopulation = SimulationMessageFactory.create('GivePopulation',
            '''A WC simulator message sent by a species pop object ...''', ['population'])

    defines a `GivePopulation` message (with an elided docstring).

    Attributes:
        __slots__ (:obj:`list`): use `__slots__` to save memory because a simulation may contain many messages
    """
    __slots__ = []

    def __init__(self, *args):
        """ Initialize a `SimulationMessage`

        `SimulationMessage` subclasses are defined by `SimulationMessageFactory.create()`.

        Args:
            args (:obj:`tuple`): argument list for initializing a subclass instance
        """
        if len(args) != len(self.__slots__):
            raise ValueError("Constructor for SimulationMessage '{}' expects {} argument(s), but "
                "{} provided".format(
                self.__class__.__name__, len(self.__slots__), len(args)))
        for slot,arg in zip(self.__slots__, args):
            setattr(self, slot, arg)

    def __str__(self):
        """ Provide a string representation of a `SimulationMessage`.
        """
        vals = []
        for attr in self.__slots__:
            if hasattr(self, attr):
                vals.append(getattr(self,attr))
            else:
                vals.append('undef')

        # TODO(Arthur): improve
        # we use str(dict()) to distinguish numeric and string attrs, but otherwise this stinks
        values = {attr:val for attr,val in zip(self.__slots__,vals)}
        return "SimulationMessage: {}({})".format(self.__class__.__name__, values)


class SimulationMessageFactory(object):
    """ Factory that creates `SimulationMessage` subclasses
    """

    @staticmethod
    def create(name, docstring, attributes=None):
        """ Compactly define a subclass of `SimulationMessage`

        To avoid confusion, the class returned by `create` should be assigned to a variable called
        `name`, which can be used to create `SimulationMessage`s of that type.

        Args:
            name (:obj:`str`): the name of the `SimulationMessage` subclass being defined
            docstring (:obj:`str`): a docstring for the subclass
            attributes (:obj:`list` of `str`, optional): attributes for the subclass which is being
                defined, if any

        Returns:
            :obj:`class`: a subclass of `SimulationMessage`

        Raises:
            :obj:`ValueError`: if `name` or `docstring` is empty
        """
        if name == '':
            raise ValueError("SimulationMessage name cannot be empty")
        if docstring == '':
            raise ValueError("SimulationMessage docstring cannot be empty")
        attrs = {}
        if attributes is not None:
            attrs['__slots__'] = attributes
        generated_simulation_message_cls = type(name, (SimulationMessage,), attrs)
        generated_simulation_message_cls.__doc__ = docstring

        return generated_simulation_message_cls
