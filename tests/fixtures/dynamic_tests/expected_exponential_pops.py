""" Expected predictions of exponential models
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-10-30
:Copyright: 2019, Karr Lab
:License: MIT
"""

#### one_rxn_exponential.xlsx ####
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
    while sum_intervals < end_time:
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

# pop_trajectory(100)


#### one_exchange_rxn_compt_growth.xlsx ####

from scipy.constants import Avogadro
# todo: obtain these constants by reading the model, & the simul params in the model
INIT_VOL_C = 1.00E-17
K_CAT = 2.50E+16
DENSITY_C = 1100
MW_A = 1.00E+06
STOCH_A = 100

END_TIME = 120
CHECKPOINT_INTERVAL = 10

# todo: calculate this
ACCOUNTED_FRACTION_C = 1.510E-01

def volume_c(pop_A,
             mw_A=MW_A,
             accounted_fraction_c=ACCOUNTED_FRACTION_C,
             density_c=DENSITY_C):
    return pop_A * mw_A / (accounted_fraction_c * density_c * Avogadro)

def rxn_rate(population,
             k_cat=K_CAT):
    return k_cat * volume_c(population)

def interval_sequence_one_exchange_rxn_compt_growth(initial_population,
                                                    k_cat=K_CAT,
                                                    end_time=END_TIME):
    # expected sequence of intervals between executions of DSA reactions
    # factor of 1/2 for steady state initial reaction timing in DSA
    initial_interval = 1 / (2 * rxn_rate(initial_population))
    intervals = [initial_interval]
    sum_intervals = initial_interval

    p = initial_population + STOCH_A
    while sum_intervals < end_time:
        interval = 1./rxn_rate(p)
        p += STOCH_A
        intervals.append(interval)
        sum_intervals += interval
    return intervals

def trajectories(initial_population,
                 initial_volume=INIT_VOL_C,
                 interval_length=CHECKPOINT_INTERVAL):
    # trajectory of population of A[c] & vol(c) at the given checkpoint intervals
    intervals = interval_sequence_one_exchange_rxn_compt_growth(initial_population)
    p = initial_population
    total_time = 0
    next_checkpoint = 0
    for interval in intervals:
        if next_checkpoint <= interval + total_time:
            print(f'{next_checkpoint}\t{p}\t{volume_c(p):.7E}')
            next_checkpoint += interval_length
        total_time += interval
        p += STOCH_A

trajectories(1000)
