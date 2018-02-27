""" Base class for simulation messages

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from abc import ABCMeta
import inspect
import warnings

from wc_sim.core.errors import SimulatorError
from wc_sim.core.utilities import ConcreteABCMeta


class SimulationMessageInterface(object, metaclass=ABCMeta):
    """ An abstract base class for simulation messages

    Each simulation event contains a simulation message. All simulation messages are objects. This
    module supports compact declaration of `SimulationMessage` types. For example::

class GivePopulation(SimulationMessage):,
            '''A WC simulator message sent by a species pop object ...''', ['population'])

    defines a `GivePopulation` message (with an elided docstring).

    Attributes:
        __slots__ (:obj:`list`): use `__slots__` to save memory because a simulation may contain many messages
    """
    __slots__ = []

    def __init__(self, *args):
        """ Initialize a `SimulationMessage`

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

        Uninitialized values are returned as `str(None)`.

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

    def value_map(self):
        """ Provide a map from attribute to value, cast to strings, for this `SimulationMessage`

        Uninitialized values are returned as `str(None)`.

        Returns:
            :obj:`dict`: map attribute -> str(attribute value)
        """
        return {attr:val for attr,val in zip(self.__slots__, self._values())}

    def values(self, annotated=False, as_list=False, separator='\t'):
        """ Provide the values in this `SimulationMessage`

        Uninitialized values are returned as `str(None)`.

        Args:
            annotated (:obj:`bool`, optional): if set, prefix each value with its attribute name
            as_list (:obj:`bool`, optional): if set, return the attribute names in a :obj:`list`
            separator (:obj:`str`, optional): the field separator used if the attribute names are returned
                as a string

        Returns:
            :obj:`obj`: `None` if this message has no attributes, or a string representation of
                the attribute names for this `SimulationMessage`, or a :obj:`list`
                representation if `as_list` is set
        """
        if not self.attrs():
            return None
        if annotated:
            list_repr = ["{}:{}".format(attr, val) for attr,val in zip(self.__slots__, self._values())]
        else:
            list_repr = self._values()
        if as_list:
            return list_repr
        else:
            return separator.join(list_repr)

    def __str__(self):
        """ Provide a string representation of a `SimulationMessage`
        """
        return "SimulationMessage: {}({})".format(self.__class__.__name__, self.value_map())

    def attrs(self):
        """ Provide a list of the attributes names for this `SimulationMessage`

        Returns:
            :obj:`list` of `str`: the attributes in this `SimulationMessage`
        """
        return self.__slots__

    def header(self, as_list=False, separator='\t'):
        """ Provide the attribute names for this `SimulationMessage`

        Args:
            as_list (:obj:`bool`, optional): if set, return the attribute names in a :obj:`list`
            separator (:obj:`str`, optional): the field separator used if the attribute names are returned
                as a string

        Returns:
            :obj:`obj`: `None` if this message has no attributes, or a string representation of
                the attribute names for this `SimulationMessage`, or a :obj:`list`
                representation if `as_list` is set
        """
        if not self.attrs():
            return None
        if as_list:
            return self.attrs()
        else:
            return separator.join(self.attrs())


class SimulationMessageMeta(type):
    # attributes mapping keyword
    ATTRIBUTES = 'attributes'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __new__(cls, clsname, superclasses, namespace):
        # Short circuit when SimulationMessage is defined
        if clsname == 'SimulationMessage':
            return super().__new__(cls, clsname, superclasses, namespace)

        if '__doc__' not in namespace:
            warnings.warn("SimulationMessage '{}' definition does not contain a docstring.".format(
                clsname))

        attrs = {}
        if cls.ATTRIBUTES in namespace:

            # check types
            attributes = namespace[cls.ATTRIBUTES]
            if not (isinstance(attributes, list) and all([isinstance(attr, str) for attr in attributes])):
                raise SimulatorError("'{}' must be a list of strings, but is '{}'".format(
                    cls.ATTRIBUTES, attributes))

            # error if attributes contains dupes
            if not len(attributes)==len(set(attributes)):
                raise SimulatorError("'{}' contains duplicates".format(cls.ATTRIBUTES))
            attrs['__slots__'] = attributes

        new_simulation_message_class = super().__new__(cls, clsname, superclasses, attrs)
        if '__doc__' in namespace:
            new_simulation_message_class.__doc__ = namespace['__doc__'].strip()
        return new_simulation_message_class

class CombinedSimulationMessageMeta(ConcreteABCMeta, SimulationMessageMeta): pass

class SimulationMessage(SimulationMessageInterface, metaclass=CombinedSimulationMessageMeta): pass
