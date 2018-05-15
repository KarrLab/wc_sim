#!/usr/bin/env python3
""" Get DataFrame of simulation checkpoint series

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-05-15
:Copyright: 2018, Karr Lab
:License: MIT
"""

import argparse
import pandas

dataframe_file = '../../wc_sim/wc_sim/multialgorithm/dataframe_file.h5'
store = pandas.HDFStore(dataframe_file)
predictions = store.get('dataframe')
print(predictions)
