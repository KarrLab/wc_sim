""" Doubly-linked list ordered by time

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-11-20
:Copyright: 2019, Karr Lab
:License: MIT
"""

import math


class DoublyLinkedNode(object):
    """ Node in doubly-linked list that stores timestamped data

    Attributes:
        _time (:obj:`Rational`): time value
        _data (:obj:`type`): the data
        _next (:obj:`DoublyLinkedNode`): next node, if linked
        _prev (:obj:`DoublyLinkedNode`): previous node, if linked
    """

    def __init__(self, time, data):
        """
        Args:
            time (:obj:`Rational`): time value
            data (:obj:`type`): the data at time `time`
        """
        self._time = time
        self._data = data
        self._next = None
        self._prev = None

    def link_on_left(self, right):
        """ Doubly-link this `DoublyLinkedNode` to the left of `right`

        Args:
            right (:obj:`DoublyLinkedNode`): a node to link on the right of this node
        """
        # use 'left' for symmetry
        left = self
        left._next = right
        right._prev = left

    def is_equal(self, other):
        """ Compare `DoublyLinkedNode`\ s

        Args:
            other (:obj:`DoublyLinkedNode`): a node to compare

        Returns:
            :obj:`bool`: `True` if `other` has the same values, otherwise `False`
        """
        if not isinstance(other, self.__class__):
            return False
        return (self._time == other._time and
                self._data == other._data)

    def __str__(self):
        """ Get a string representation

        Returns:
            :obj:`str`: string representation of a `TimeOrderedList`
        """
        rv = []
        for attr in ['_time', '_data']:
            rv.append(f"{attr}: {getattr(self, attr)}")
        if self._prev is None:
            rv.append('_prev is None')
        else:
            rv.append('_prev is not None')
        if self._next is None:
            rv.append('_next is None')
        else:
            rv.append('_next is not None')
        return '\n'.join(rv)


class TimeOrderedList(object):
    """ Doubly-linked list ordered by time

    Attributes:
        _head (:obj:`DoublyLinkedNode`): head node, which always has time = -infinity
        _tail (:obj:`DoublyLinkedNode`): head node, which always has time = +infinity
        _count (:obj:`int`): number of data nodes in this list
    """

    ### internal methods ###
    # be careful using internal methods -- they don't check bad input
    def __init__(self):
        # -inf and +inf sentinels simplify many operations
        self._head = DoublyLinkedNode(float("-inf"), None)
        self._tail = DoublyLinkedNode(float("inf"), None)
        self._head.link_on_left(self._tail)
        self._count = 0

    def _insert_after(self, existing_node, new_node):
        """ Insert `new_node` after node `existing_node`

        Args:
            existing_node (:obj:`DoublyLinkedNode`): node in the doubly-linked list
            new_node (:obj:`DoublyLinkedNode`): node to insert to the right of `existing_node`
        """
        # link new_node in between existing_node and the node on its right
        right_of_new_node = existing_node._next
        existing_node.link_on_left(new_node)
        new_node.link_on_left(right_of_new_node)
        self._count += 1

    def _delete(self, existing_node):
        """ Delete node `existing_node`

        Args:
            existing_node (:obj:`DoublyLinkedNode`): node in the doubly-linked list

        Returns:
            :obj:`DoublyLinkedNode`: the node that was existed, or `None` if the list is empty
        """
        if self.is_empty():
            return None
        # splice out existing_node
        existing_node._prev.link_on_left(existing_node._next)
        self._count -= 1
        return existing_node

    def _check_time(self, method, time):
        """ Check that `time` is a good time

        Args:
            method (:obj:`str`): the calling method that's checking the time
            time (:obj:`Rational`): time value

        Raises:
            :obj:`ValueError`: if `time` isn't an :obj:`int` or :obj:`float`, or is infinite or `NaN`
        """
        # check the time for method
        if not isinstance(time, (int, float)):
            raise ValueError(f"time '{time}' isn't a number")
        if math.isinf(time) or math.isnan(time):
            raise ValueError(f"cannot {method} at time {time}")

    ### external methods ###
    def is_empty(self):
        """ Is the sorted list empty?

        Returns:
            :obj:`bool`: `True` if the list is empty, otherwise `False`
        """
        return self._count == 0

    def clear(self):
        """ Clear this sorted list

        Remove all data nodes.
        """
        self._head.link_on_left(self._tail)
        self._count = 0

    def len(self):
        """ Provide length of this sorted list

        Returns:
            :obj:`int`: number of data nodes in this list
        """
        return self._count

    def insert(self, time, data):
        """ Insert data `data` at time `time`

        Args:
            time (:obj:`Rational`): time value
            data (:obj:`type`): the data at time `time`

        Raises:
            :obj:`ValueError`: if data at time is already in the list
        """
        self._check_time('insert', time)
        left_node = self.find(time)
        if left_node._time == time:
            raise ValueError(f"time {time} already in queue")
        else:
            self._insert_after(left_node, DoublyLinkedNode(time, data))

    def find(self, time):
        """ Find the node with the largest time <= `time`

        Args:
            time (:obj:`Rational`): time value

        Returns:
            :obj:`DoublyLinkedNode`: node with the largest time <= `time`
        """
        self._check_time('find', time)
        node = self._head
        while node._time <= time:
            node = node._next
        return node._prev

    def delete(self, time):
        """ Delete the node at time `time`

        Args:
            time (:obj:`Rational`): time value

        Returns:
            :obj:`DoublyLinkedNode`: the node that's deleted

        Raises:
            :obj:`ValueError`: if no node with time `time` is in the queue
        """
        self._check_time('delete', time)
        left_node = self.find(time)
        if left_node._time == time:
            return self._delete(left_node)
        else:
            raise ValueError(f"no node with time {time} in queue")

    def gc_old_data(self, time, min_to_keep=2):
        """ Garbage collect old nodes and data

        Keep all nodes with time >= `time` and at least `min_to_keep` of the highest time nodes, if
        enough are available. Delete the rest. If the queue has fewer than `min_to_keep` nodes do nothing.
        Recommendation: keep `min_to_keep` large enough to enable any fitting done over nearby data.

        Args:
            time (:obj:`Rational`): time value
            min_to_keep (:obj:`int`): minimum number of nodes to leave in this `TimeOrderedList`

        Returns:
            :obj:`int`: the number of nodes deleted
        """
        self._check_time('gc_old_data', time)
        if self.len() < min_to_keep or self.len() == 0:
            return 0
        node = self._tail._prev
        num_keeping = 0
        # iterate down the list to the highest time node to delete
        while time <= node._time:
            node = node._prev
            num_keeping += 1
        if node == self._head:
            return 0
        # iterate further if not keeping enough, but don't go to the head
        while num_keeping < min_to_keep and node._prev and node._prev._prev:
            node = node._prev
            num_keeping += 1
        highest_to_delete = node
        num_being_deleted = self.len() - num_keeping
        self._count = num_keeping
        # splice around all nodes between head and the node to the right of highest_to_delete
        self._head.link_on_left(highest_to_delete._next)
        return num_being_deleted

    def __str__(self):
        """ Get a string representation

        Returns:
            :obj:`str`: string representation of a `TimeOrderedList`
        """
        rv = []
        rv.append(f'{self.len()} nodes:')
        node = self._head._next
        while node._time < float('inf'):
            rv.append(f'{node._time}: {node._data}')
            node = node._next
        return '\n'.join(rv)


class LinearInterpolationList(TimeOrderedList):
    """ Time ordered list with linear interpolation of data values in nodes
    """

    def read(self, time):
        """ Get the data at time `time`, interpolating if needed

        Linear interpolation is done when the time is between data points.

        Args:
            time (:obj:`Rational`): time value

        Returns:
            :obj:`type`: the estimated data at `time`, or `NaN` if it cannot be estimated
        """
        self._check_time('read', time)
        left_node = self.find(time)
        if left_node._time == time:
            return left_node._data
        # interpolate if possible
        if left_node._next._time == float("inf"):
            # cannot interpolate with no data at times greater than `time` - assume slope == 0
            return left_node._data
        elif left_node._time == -float("inf"):
            # no data available at time <= `time`
            return float('NaN')
        else:
            # get slope between left_node and left_node._next
            slope = (left_node._next._data - left_node._data) / (left_node._next._time - left_node._time)
        # interpolate
        return left_node._data + slope * (time - left_node._time)


class MultiAlgorithmSpeciesPopNode(object):
    """ Data node in a `MultiAlgorithmSpeciesPopHistory`

    Attributes:
        pop_before (:obj:`float`): species population infinitesimally before the time of this node
        pop_after (:obj:`float`): species population infinitesimally after the time of this node
    """

    def __init__(self,  **kwargs):
        """
        Args:
            **kwargs (:obj:`dict`, optional): keywords for initialization
                allowed keywords:
                    `pop_before` - if supplied, will be assigned to `self.pop_before`
                    `pop_after` - if supplied, will be assigned to `self.pop_after`
                    `pop` - if supplied, will be assigned to `self.pop_before` and `self.pop_after`
                        if they're not otherwise initialized
        """
        kwargs_mapping = (('pop', 'pop_before'),
                          ('pop', 'pop_after'),
                          ('pop_before', 'pop_before'),
                          ('pop_after', 'pop_after'),
                         )
        self.pop_before = None
        self.pop_after = None
        for keyword, attr_name in kwargs_mapping:
            if keyword in kwargs:
                setattr(self, attr_name, kwargs[keyword])


class MultiAlgorithmSpeciesPopHistory(TimeOrderedList):
    """ Time ordered list with linear interpolation of data values in nodes

    Manage a species population history with step function changes from discrete time
    integration algorithms like SSA, and slope changes from continuous time integration
    algorithms like ODE and dFBA.

    To enable interpolation of population, a `MultiAlgorithmSpeciesPopHistory` must be initialized
    with 1 data node, and never contain fewer than 2 data nodes after a 2nd one is inserted.
    """

    @property
    def _earliest_time(self):
        """ The time of the earliest data in the history
        """
        return self._head._next._time

    @property
    def _latest_time(self):
        """ The time of the latest data in the history
        """
        return self._tail._prev._time

    def _check_time_strict(self, time):
        """ Check that `time` is a good time

        Args:
            time (:obj:`Rational`): time value

        Raises:
            :obj:`ValueError`: if `time` is earlier than the earliest data in the history, or
                later than the latest data in the history
        """
        if time < self._earliest_time:
            raise ValueError(f"time {time} earlier than earliest data {self._earliest_time} in history")
        if self._latest_time < time:
            raise ValueError(f"time {time} later than the latest data {self._latest_time} in history")

    def _slope_between_node_pair(self, left_node):
        """ Get the population slope between `left_node` and the node to its right

        Args:
            left_node (:obj:`DoublyLinkedNode`): node in the history

        Returns:
            :obj:`float`: the rate of population change between `left_node` and the node to its right
        """
        # todo: perhaps raise an error if pops aren't defined or times are infinite
        right_node = left_node._next
        return (right_node._data.pop_before - left_node._data.pop_after) / \
               (right_node._time - left_node._time)

    def slope(self, time, left_node=None):
        """ Get the population slope at time `time`

        Args:
            time (:obj:`Rational`): time value
            left_node (:obj:`DoublyLinkedNode`, optional): node to use to avoid cost of `find`

        Returns:
            :obj:`float`: the estimated population slope at `time`

        Raises:
            :obj:`ValueError`: if `time` is earlier than the earliest data in the history, or
                later than the latest data
        """
        self._check_time('slope', time)
        if self._latest_time < time:
            # slope unknown for time later than latest data in history
            return 0
        self._check_time_strict('slope', time)
        if left_node is None:
            left_node = self.find(time)
        if left_node._time == time:
            if left_node._time == self._earliest_time and left_node._time == self._latest_time:
                # only 1 node in history; cannot compute slope
                return 0
            if left_node._time == self._earliest_time:
                # at least 1 node in history after left_node; return slope between left and next node
                return self._slope_between_node_pair(left_node)
            if left_node._time == self._latest_time:
                # at least 1 node in history before left_node; return slope between prev node and left node
                return self._slope_between_node_pair(left_node._prev)
            # history contains at least 1 node before left_node and 1 node after left_node
            # symmetrically, return slope between prev node and next node
            prev_node = left_node._prev
            next_node = left_node._next
            return (next_node._data.pop_before - prev_node._data.pop_after) / \
                   (next_node._time - next_node._time)

        else:
            # at least 1 node in history after left_node; return slope between left and next node
            return self._slope_between_node_pair(left_node)

    def read(self, time, left_node=None):
        """ Get the data at time `time`, interpolating if needed

        By convention, if time `time` coincides with a discrete population change, then `read`
            returns `pop_before`, the species population infinitesimally before the time
        Linear interpolation is done when adjacent data points are available.

        Args:
            time (:obj:`Rational`): time value
            left_node (:obj:`DoublyLinkedNode`, optional): node to use to avoid cost of `find`

        Returns:
            :obj:`tuple`: the estimated population and slope at `time`
        """
        self._check_time('read', time)
        self._check_time_strict('read', time)
        if left_node is None:
            left_node = self.find(time)
        # interpolate
        slope = self.slope(time, left_node=left_node)
        if left_node._time == time:
            return (left_node._data.pop_before, slope)
        return (slope * (time - left_node._time) + left_node._data.pop_after, slope)

    def insert_discrete_change(self, time, population_change):
        """ Insert a discrete population change at time `time`

        Args:
            time (:obj:`Rational`): time value
            population_change (:obj:`float`): the population change at time `time`
        """
        self._check_time('insert_discrete_change', time)
        left_node = self.find(time)
        if left_node._time != time:
            # if no change exists at time, so add one
            pop, _ = self.read(time, left_node=left_node)
            # change in population is additive
            pop_after = pop + population_change
            kwargs = dict(pop_before=pop,
                          pop_after=pop_after)
            data_node = MultiAlgorithmSpeciesPopNode(**kwargs)
            self._insert_after(left_node, DoublyLinkedNode(time, data_node))
        else:
            # else update the existing change at time
            data_node = left_node.data
            # change in population is additive
            data_node.pop_after = data_node.pop_after + population_change

        # propagate the change
        self.propagate_change(time)

    def insert_continuous_change(self, start_time, end_time, end_population):
        """ Insert a continuous population change that ends at time `end_time`

        Args:
            start_time (:obj:`Rational`): time when the continuous population change started
            end_time (:obj:`Rational`): time when the continuous population change ended
            end_population (:obj:`float`): the species population at time `end_time`

        Raises:
            :obj:`ValueError`: if `end_time` <= `start_time`
        """
        self._check_time('insert_continuous_change: start_time', start_time)
        self._check_time('insert_continuous_change: end_time', end_time)
        if end_time <= start_time:
            raise ValueError(f"end_time {end_time} <= start_time {start_time}")

        # slope of change = (end_population - pop @ end_time) / (end_time - start_time)
        pop_at_end_time, _ = self.read(end_time)
        slope_of_change = (end_population - pop_at_end_time) / (end_time - start_time)
        if slope_of_change == 0:
            return

        # the slope of change is recorded by changing populations in [start_time, end_time]:
        # 1) a node at `start_time` records the current population there
        # 2) all nodes in (start_time, end_time) have their populations changed to reflect the slope
        # 3) a node at `end_time` has pop_before = end_population

        # propagate the change
        self.propagate_change(time, duration=duration)

    def propagate_change(self, time):
        """ Propagate a change made at time `time` to the end of this history

        Args:
            time (:obj:`Rational`): time value
        """
        # todo: write

    def average_pop(self, start_time, end_time):
        """ Determine the average population from `start_time` to `end_time`

        Args:
            start_time (:obj:`Rational`): start time of the average
            end_time (:obj:`Rational`): end time of the average
        """
        # todo: write
