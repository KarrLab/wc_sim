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
        self._temp_data = None
        self._next = None
        self._prev = None
    def connect_right(self, right):
        # doubly-link with node on right
        # use 'left' for symmetry
        left = self
        left._next = right
        right._prev = left
    def __str__(self):
        rv = []
        for attr in ['_time', '_data', '_temp_data']:
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
    ### external methods ###
    def is_empty(self):
        return self._count == 0
    def clear(self):
        self._head.connect_right(self._tail)
        self._count = 0
    def len(self):
        return self._count
    def insert(self, time, value):
        # insert value at time
        if math.isinf(time) or math.isnan(time):
            raise ValueError(f"cannot insert at time {time}")
        left_node = self.find(time)
        if left_node._time == time:
            raise ValueError(f"time {time} already in queue")
        else:
            self._insert_after(left_node, Node(time, value))
    def find(self, time):
        # find node with largest time <= time
        if not isinstance(time, (int, float)):
            raise ValueError(f"time {time} isn't a number")
        if time == float("inf"):
            return self._tail
        node = self._head
        while node._time <= time:
            node = node._next
        return node._prev
    def delete(self, time):
        # delete node at time
        left_node = self.find(time)
        if left_node._time == time:
            return self._delete(left_node)
        else:
            raise ValueError(f"no node with time {time} in queue")
    def read(self, time):
        # get value at time, interpolating if needed
        if math.isinf(time) or math.isnan(time):
            raise ValueError(f"cannot read at time {time}")
        left_node = self.find(time)
        if left_node._time == time:
            return left_node.data
        # interpolate if possible
        if self.len() == 1:
            # with only one data point cannot interpolate
            return left_node.data
        elif left_node._next._time == float("inf"):
            # get slope between left_node._prev and left_node
            slope = (left_node.data - left_node._prev.data) / (left_node._time - left_node._prev._time)
        else:
            # get slope between left_node and left_node._next
            slope = (left_node._next.data - left_node.data) / (left_node._next._time - left_node._time)
        # interpolate
        return slope * (time - left_node._time)
    def gc(self, time):
        # delete all nodes with time < time
        node = self._head._next
        while node._time < time:
            next_node = node._next
            self._delete(node)
            node = next_node
    def temp_insert(self, time, value):
        # temporarily insert value at time -- allows time collisions
        if math.isinf(time) or math.isnan(time):
            raise ValueError(f"cannot temporarily insert at time {time}")
        left_node = self.find(time)
        if left_node._time == time:
            left_node._temp_data = value
        else:
            self._insert_after(left_node, Node(time, value))
    def temp_delete(self, time):
        # delete temporarily inserted value at time
        if math.isinf(time) or math.isnan(time):
            raise ValueError(f"cannot delete temporary insert at time {time}")
        left_node = self.find(time)
        if left_node._time != time:
            raise ValueError(f"cannot find temporarily inserted value at time {time}")
        else:
            if left_node._temp_data is None:
                return self._delete(left_node)
            else:
                left_node._temp_data = None
