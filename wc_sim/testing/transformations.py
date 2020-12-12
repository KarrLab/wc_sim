""" Model transformations for testing

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-12-11
:Copyright: 2020, Karr Lab
:License: MIT
"""

import obj_tables
import wc_lang


class SetStdDevsToZero(wc_lang.transform.Transform):
    """ Set standard deviations of data in a whole-cell model used by a simulation to 0 """

    class Meta(object):
        id = 'SetStdDevsToZero'
        label = 'Set standard deviations of data in a whole-cell model used by a simulation to 0'

    def run(self, wc_model):
        """ Set standard deviations of data in a whole-cell model used by a simulation to 0

        Set standard deviations of data used by a simulation to 0: initial volumes, distributions of initial
        concentrations, and parameters

        Args:
            wc_model (:obj:`wc_lang.Model`): a whole-cell model

        Returns:
            :obj:`Model`: `wc_model`, with standard deviations of data used by a simulation set to 0
        """
        models_with_std_devs = (wc_lang.InitVolume,
                                wc_lang.DistributionInitConcentration,
                                wc_lang.Parameter)
        # assumes that standard deviation attributes are named 'std', the current convention
        for model_instance in obj_tables.Model.get_all_related([wc_model]):
            if isinstance(model_instance, models_with_std_devs):
                model_instance.std = 0
