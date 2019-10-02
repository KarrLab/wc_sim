"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-10-02
:Copyright: 2016-2019, Karr Lab
:License: MIT
"""


class TestMultialgorithmSimulation(unittest.TestCase):

    def test_partition_species(self):
        self.multialgorithm_simulation.partition_species()
        priv_species = self.multialgorithm_simulation.private_species
        for key, val in priv_species.items():
            priv_species[key] = set(val)
        expected_priv_species = dict(
            submodel_1=set(['species_1[e]', 'species_2[e]', 'species_1[c]']),
            submodel_2=set(['species_5[c]', 'species_6[c]'])
        )
        self.assertEqual(priv_species, expected_priv_species)
        expected_shared_species = set(['species_2[c]', 'species_3[c]', 'species_4[c]', 'H2O[e]', 'H2O[c]'])
        self.assertEqual(self.multialgorithm_simulation.shared_species, expected_shared_species)
