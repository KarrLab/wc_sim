""" Base class for simulation messages

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import six, abc

from wc_sim.core.errors import SimulatorError


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

        Raises:
            :obj:`SimulatorError`: if `args` does not contain an argument for each entry in __slots__
        """
        if len(args) != len(self.__slots__):
            raise SimulatorError("Constructor for SimulationMessage '{}' expects {} argument(s), but "
                "{} provided".format(
                self.__class__.__name__, len(self.__slots__), len(args)))
        for slot,arg in zip(self.__slots__, args):
            setattr(self, slot, arg)

    def _values(self):
        """ Provide the values in a `SimulationMessage`, cast to strings

        Returns:
            :obj:`list`: list of attribute values
        """
        vals = []
        for attr in self.__slots__:
            if hasattr(self, attr):
                vals.append(str(getattr(self,attr)))
            else:
                vals.append(str(None))
        return vals

    def values(self):
        """ Provide the values in a `SimulationMessage`, cast to strings

        Returns:
            :obj:`dict`: map attribute -> str(attribute value)
        """
        return {attr:val for attr,val in zip(self.__slots__,self._values())}

    def __str__(self):
        """ Provide a string representation of a `SimulationMessage`
        """
        return "SimulationMessage: {}({})".format(self.__class__.__name__, self.values())

    def attrs(self):
        """ Provide a list of the attributes names for this `SimulationMessage`

        Returns:
            :obj:`list` of `str`: the attributes in this `SimulationMessage`
        """
        return self.__slots__

    def header(self):
        """ Provide a tab-delimited string of the attribute names for this `SimulationMessage`

        Returns:
            :obj:``str`: the attributes in this `SimulationMessage`, tab-delimited, or `None`
                if this message has no attributes
        """
        if self.attrs():
            return '\t'.join(self.attrs())
        else:
            return None

    def delimited(self):
        """ Provide a tab-delimited string containing values in this `SimulationMessage`

        Returns:
            :obj:``str`: the values of the attributes in this `SimulationMessage`, tab-delimited
        """
        return '\t'.join(self._values())


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
            :obj:`SimulatorError`: if `name` or `docstring` is empty
        """
        if name == '':
            raise SimulatorError("SimulationMessage name cannot be empty")
        if docstring == '':
            raise SimulatorError("SimulationMessage docstring cannot be empty")
        attrs = {}
        if attributes is not None:
            attrs['__slots__'] = attributes
        generated_simulation_message_cls = type(name, (SimulationMessage,), attrs)
        generated_simulation_message_cls.__doc__ = docstring.strip()

        return generated_simulation_message_cls
