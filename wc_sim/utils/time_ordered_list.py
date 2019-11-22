""" Doubly-linked list ordered by time

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-11-20
:Copyright: 2019, Karr Lab
:License: MIT
"""

import math


class Node(object):

    def __init__(self, time, data):
        self._time = time
        self._data = data
        self._next = None
        self._prev = None

    def connect_right(self, right):
        # doubly-link with node on right
        # use 'left' for symmetry
        left = self
        left._next = right
        right._prev = left

    def is_equal(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self._time == other._time and
                self._data == other._data)

    def __str__(self):
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
    # Doubly-linked list ordered by time

    ### internal methods ###
    # be careful using internal methods -- they don't check bad input
    def __init__(self):
        # -inf and +inf sentinels simplify most operations
        self._head = Node(float("-inf"), None)
        self._tail = Node(float("inf"), None)
        self._head.connect_right(self._tail)
        self._count = 0

    def _insert_after(self, existing_node, new_node):
        # insert after node `existing_node`
        # patch new_node in
        right_of_new_node = existing_node._next
        existing_node.connect_right(new_node)
        new_node.connect_right(right_of_new_node)
        self._count += 1

    def _delete(self, existing_node):
        if self.is_empty():
            return None
        # connect around existing_node
        existing_node._prev.connect_right(existing_node._next)
        self._count -= 1
        return existing_node

    def _check_time(self, method, time):
        # check the time for method
        if not isinstance(time, (int, float)):
            raise ValueError(f"time '{time}' isn't a number")
        if math.isinf(time) or math.isnan(time):
            raise ValueError(f"cannot {method} at time {time}")

    ### external methods ###
    def is_empty(self):
        return self._count == 0

    def clear(self):
        # remove references to the data nodes so they can be gc'ed
        self._head.connect_right(self._tail)
        self._count = 0

    def len(self):
        return self._count

    def insert(self, time, value):
        # insert value at time
        self._check_time('insert', time)
        left_node = self.find(time)
        if left_node._time == time:
            raise ValueError(f"time {time} already in queue")
        else:
            self._insert_after(left_node, Node(time, value))

    def find(self, time):
        # find node with largest time <= time
        self._check_time('find', time)
        node = self._head
        while node._time <= time:
            node = node._next
        return node._prev

    def delete(self, time):
        # delete node at time
        self._check_time('delete', time)
        left_node = self.find(time)
        if left_node._time == time:
            return self._delete(left_node)
        else:
            raise ValueError(f"no node with time {time} in queue")

    def read(self, time):
        # get value at time, interpolating if needed
        self._check_time('read', time)
        left_node = self.find(time)
        if left_node._time == time:
            return left_node._data
        # interpolate if possible
        if left_node._next._time == float("inf"):
            # cannot interpolate with no data at time greater than time - assume slope == 0
            return left_node._data
        elif left_node._time == -float("inf"):
            # no data available at time <= `time`
            return float('NaN')
        else:
            # get slope between left_node and left_node._next
            slope = (left_node._next._data - left_node._data) / (left_node._next._time - left_node._time)
        # interpolate
        return left_node._data + slope * (time - left_node._time)

    def gc(self, time, num_to_leave=2):
        # delete all nodes with time < time, leaving at least num_to_leave
        # to enable interpolation, keep 2 <= num_to_leave
        # todo: make fast by searching from right end & removing all nodes to left of first
        # node to keep by splicing them out
        node = self._head._next
        while node._time < time and num_to_leave < self.len():
            next_node = node._next
            self._delete(node)
            node = next_node

    def __str__(self):
        rv = []
        rv.append(f'{self.len()} nodes:')
        node = self._head._next
        while node._time < float('inf'):
            rv.append(f'{node._time}: {node._data}')
            node = node._next
        return '\n'.join(rv)
