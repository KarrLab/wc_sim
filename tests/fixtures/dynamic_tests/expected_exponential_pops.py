""" Expected population for one_rxn_exponential.xlsx
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-10-30
:Copyright: 2019, Karr Lab
:License: MIT
"""

CHECKPOINT_INTERVAL = 10
K_CAT = 0.04
END_TIME = 50

def interval_sequence(initial_population,
                      k_cat=K_CAT,
                      end_time=END_TIME):
    # expected sequence of intervals between executions of DSA reactions
    p = initial_population
    # factor of 1/2 for short initial reaction in DSA
    initial_interval = 1./(2*k_cat*p)
    p += 1
    intervals = [initial_interval]
    sum_intervals = initial_interval
    while sum_intervals <= end_time:
        interval = 1./(k_cat*p)
        p += 1
        intervals.append(interval)
        sum_intervals += interval
    return intervals

def pop_trajectory(initial_population,
                   interval_length=CHECKPOINT_INTERVAL):
    # trajectory of population at the given checkpoint intervals
    intervals = interval_sequence(initial_population)
    p = initial_population
    total_time = 0
    next_checkpoint = 0
    for interval in intervals:
        if next_checkpoint <= interval + total_time:
            print(f'{next_checkpoint}\t{p}')
            next_checkpoint += interval_length
        total_time += interval
        # print(f"{total_time:.5f}")
        p += 1

pop_trajectory(100)
