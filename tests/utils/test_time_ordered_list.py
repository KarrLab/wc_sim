"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-11-21   
:Copyright: 2019, Karr Lab
:License: MIT
"""

import math
import numpy
import random
import timeit
import unicodedata
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

        # test connect_right
        n.connect_right(o)
        is_connected(self, n, o)
        self.assertIn('is not None', str(n))
        self.assertIn('is not None', str(o))

        # test is_equal
        p = Node(4, 'y')
        self.assertTrue(p.is_equal(p))
        self.assertTrue(p.is_equal(o))
        self.assertTrue(o.is_equal(p))
        p._time += 1
        self.assertTrue(p.is_equal(p))
        self.assertFalse(p.is_equal(o))
        self.assertFalse(p.is_equal('hi'))

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

        ### test internal methods ###
        tol = TimeOrderedList()
        invariants(self, tol)

        # test _insert_after
        value = 'x'
        n = Node(1, value)
        tol._insert_after(tol._head, n)
        self.assertEqual(tol.len(), 1)
        invariants(self, tol)
        self.assertEqual(tol._head._next._data, value)
        self.assertEqual(tol._tail._prev._data, value)

        self.assertIn('nodes:', str(tol))
        self.assertIn(value, str(tol))

        # test _delete
        self.assertEqual(tol._delete(n), n)
        invariants(self, tol)
        self.assertTrue(tol.is_empty())

        self.assertEqual(tol._delete(n), None)
        invariants(self, tol)

        # test _check_time
        bad_time = 'x'
        with self.assertRaisesRegex(ValueError, "time '.*' isn't a number"):
            tol._check_time('method', bad_time)
        bad_time = float('inf')
        with self.assertRaisesRegex(ValueError, 'cannot .* at time'):
            tol._check_time('method', bad_time)
        bad_time = float('NaN')
        with self.assertRaisesRegex(ValueError, 'cannot .* at time'):
            tol._check_time('method', bad_time)
        good_time = 3
        self.assertEqual(tol._check_time('method', good_time), None)
        good_time = 1.34e4
        self.assertEqual(tol._check_time('method', good_time), None)

        ### test external methods ###
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
        n = tol.find(max(times) + 1)
        self.assertEqual(n._time, max(times))
        n = tol.find(min(times) - 1)
        self.assertEqual(n._time, -float('inf'))
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
        
        # test gc
        tol = TimeOrderedList()
        maximum = 10
        for t in reversed(range(maximum)):
            tol.insert(t, t)
        invariants(self, tol)
        self.assertEqual(tol.len(), maximum)
        tol.gc(4.001)
        invariants(self, tol)
        self.assertEqual(tol.len(), 5)
        tol.gc(8)
        invariants(self, tol)
        self.assertEqual(tol.len(), 2)
        num_to_leave = 1
        tol.gc(11, num_to_leave=num_to_leave)
        invariants(self, tol)
        self.assertEqual(tol.len(), num_to_leave)

        # test read
        tol = TimeOrderedList()
        first_2_values = 5
        last_value = 10
        time_n_values = [(1, first_2_values), (2, first_2_values), (3, last_value), ]
        for time, value in time_n_values:
            tol.insert(time, value)
        invariants(self, tol)
        for time, value in time_n_values:
            self.assertEqual(tol.read(time), value)
        self.assertTrue(math.isnan(tol.read(0.5)))
        self.assertEqual(tol.read(1.01), first_2_values)
        self.assertEqual(tol.read(1.5), first_2_values)
        self.assertEqual(tol.read(2.5), 0.5*(first_2_values + last_value))
        self.assertEqual(tol.read(3.5), last_value)

    def test_time_ordered_list_performance(self):
        # measure the performance of TimeOrderedList.find()
        print()
        def measure_time_ordered_list_performance(num_items, timeit_num=100, num_measurements=100):
            tol = TimeOrderedList()
            value = 'x'
            # reverse to build rapidly
            for time in reversed(range(num_items)):
                tol.insert(time, value)
            find_times = []
            for _ in range(num_measurements):
                time = num_items * random.random()
                # find is the expensive method
                find_expr = f'tol.find({time})'
                m = timeit.timeit(find_expr, number=timeit_num, globals=locals())
                find_times.append(m)
            factor = 1_000_000 / timeit_num
            # time in microsec
            mean_time = factor * numpy.mean(find_times)
            print(f"{num_items}\t{mean_time:.2f}")

        mu = unicodedata.lookup('GREEK SMALL LETTER MU')
        print(f"# TOL items\tE[time(find)] ({mu}sec)")
        factors_of_ten = 4
        all_num_items = [10 ** factor_of_ten for factor_of_ten in range(factors_of_ten)]
        for num_items in all_num_items:
            measure_time_ordered_list_performance(num_items)
