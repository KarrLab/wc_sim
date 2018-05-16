#!/usr/bin/env python3
""" Get DataFrame of simulation checkpoint series

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-05-15
:Copyright: 2018, Karr Lab
:License: MIT
"""

import pandas

# TODO(Arthur): make general purpose
dataframe_file = '/Users/arthur_at_sinai/gitOnMyLaptopLocal/wc_sim/wc_sim/multialgorithm/dataframe_file.h5'
store = pandas.HDFStore(dataframe_file)
predictions = store.get('dataframe')
print(predictions)
store.close()
