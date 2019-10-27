""" Utilities for testing

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-10-26
:Copyright: 2019, Karr Lab
:License: MIT
"""

from wc_lang import Model
from wc_lang.io import Reader
import wc_lang


def read_model_and_set_all_std_devs_to_0(model_filename):
    """ Read a model and set all standard deviations to 0

    Args:
        model_filename (:obj:`str`): `wc_lang` model file

    Returns:
        :obj:`Model`: a whole-cell model
    """
    # read model while ignoring missing models
    data = Reader().run(model_filename, ignore_extra_models=True)
    # set all standard deviations to 0
    models_with_std_devs = (wc_lang.InitVolume,
                            wc_lang.Ph,
                            wc_lang.DistributionInitConcentration,
                            wc_lang.Parameter,
                            wc_lang.Observation,
                            wc_lang.Conclusion)
    for model, instances in data.items():
        if model in models_with_std_devs:
            for instance in instances:
                instance.std = 0
    return data[Model][0]
