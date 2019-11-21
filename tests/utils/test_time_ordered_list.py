"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-11-21   
:Copyright: 2019, Karr Lab
:License: MIT
"""

import unittest

from wc_sim.utils.time_ordered_list import Node, TimeOrderedList


class Test(unittest.TestCase):

    def test_node(self):

        def is_connected(test_case, left, right):
            test_case.assertEqual(left._next._prev, left)
            test_case.assertEqual(right._prev._next, right)
            test_case.assertEqual(left._next, right)
            test_case.assertEqual(left, right._prev)

        n = Node(3, 'x')
        self.assertIn('_time', str(n))
        self.assertIn('_data', str(n))
        self.assertIn('is None', str(n))
        o = Node(4, 'y')
        n.connect_right(o)
        is_connected(self, n, o)
        self.assertIn('is not None', str(n))
        self.assertIn('is not None', str(o))

    def test_time_ordered_list(self):

        def invariants(test_case, time_ordered_list):
            # head tail invariant
            test_case.assertEqual(time_ordered_list._head._next._prev, time_ordered_list._head)
            test_case.assertEqual(time_ordered_list._tail._prev._next, time_ordered_list._tail)

            # ordered invariant
            node = time_ordered_list._head
            while node._next is not None:
                self.assertTrue(node._time < node._next._time)
                node = node._next

        # test internal methods
        tol = TimeOrderedList()
        invariants(self, tol)

        value = 'x'
        n = Node(1, value)
        tol._insert_after(tol._head, n)
        self.assertEqual(tol.len(), 1)
        invariants(self, tol)
        self.assertEqual(tol._head._next._data, value)
        self.assertEqual(tol._tail._prev._data, value)

        self.assertEqual(tol._delete(n), n)
        invariants(self, tol)
        self.assertTrue(tol.is_empty())

        self.assertEqual(tol._delete(n), None)
        invariants(self, tol)

        # test external methods
        tol = TimeOrderedList()
        self.assertTrue(tol.is_empty())
        self.assertEqual(tol.len(), 0)
        time = 1
        tol.insert(time, 3)
        invariants(self, tol)

        tol.clear()
        self.assertTrue(tol.is_empty())
        invariants(self, tol)

        # test insert
        value = 'z'
        tol.insert(time, value)
        invariants(self, tol)

        with self.assertRaisesRegex(ValueError, 'cannot insert at time'):
            tol.insert(float("inf"), value)
        with self.assertRaisesRegex(ValueError, 'cannot insert at time'):
            tol.insert(float("NaN"), value)
        time = 3
        tol.insert(time, value)
        with self.assertRaisesRegex(ValueError, "time .* already in queue"):
            tol.insert(time, value)

        times = [4, -3, 2, 10, 6]
        for time in times:
            tol.insert(time, time)
        invariants(self, tol)

        # test find
        for time in times:
            n = tol.find(time)
            self.assertEqual(n._data, time)
        self.assertEqual(tol.find(float("-inf")), tol._head)
        self.assertEqual(tol.find(float("inf")), tol._tail)
        with self.assertRaisesRegex(ValueError, "time .* isn't a number"):
            tol.find('h')

        # test delete
        len = tol.len()
        for time in times:
            n = tol.delete(time)
            self.assertTrue(isinstance(n, Node))
            self.assertEqual(tol.len(), len - 1)
            len -= 1
        with self.assertRaisesRegex(ValueError, "no node with time .* in queue"):
            tol.delete(times[0])
        
