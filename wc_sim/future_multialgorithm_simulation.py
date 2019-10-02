""" Future improvements to wc_sim/multialgorithm_simulation.py; useful later

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-10-02
:Copyright: 2016-2019, Karr Lab
:License: MIT
"""

DEFAULT_VALUES = dict(
    shared_species_store='SHARED_SPECIE_STORE'
)


class MultialgorithmSimulation(object):
    """ Initialize a multialgorithm simulation from a language model and run-time parameters

    Attributes:
        shared_species_store_name (:obj:`str`): the name for the shared specie store
        species_pop_objs (:obj:`dict` of `SpeciesPopSimObject`): shared species
            populations used by `SimulationObject`\ 's; not currently used
        private_species (:obj:`dict` of `set`): map from `DynamicSubmodel` to a set of the species
                modeled by only the submodel; not currently used
        shared_species (:obj:`set`): the shared species
    """

    def __init__(self, model, args, shared_species_store_name=DEFAULT_VALUES['shared_species_store']):
        """
        Args:
            shared_species_store_name (:obj:`str`, optional): the name of the shared species store; not currently used
        """
        self.shared_species_store_name = shared_species_store_name  # not currently used
        self.species_pop_objs = {}

    # not currently used
    def partition_species(self):
        """ Partition species populations for this model's submodels
        """
        self.init_populations = self.get_initial_species_pop(self.model, self.random_state)
        # self.private_species not currently used
        self.private_species = ModelUtilities.find_private_species(self.model, return_ids=True)
        self.shared_species = ModelUtilities.find_shared_species(self.model, return_ids=True)
        # self.species_pop_objs not currently used
        self.species_pop_objs = self.create_shared_species_pop_objs()

    # not currently used
    def create_shared_species_pop_objs(self):
        """ Create the shared species object.

        Returns:
            dict: `dict` mapping id to `SpeciesPopSimObject` objects for the simulation
        """
        species_pop_sim_obj = SpeciesPopSimObject(
            self.shared_species_store_name,
            {species_id: self.init_populations[species_id] for species_id in self.shared_species},
            molecular_weights=self.molecular_weights_for_species(self.shared_species),
            random_state=self.random_state)
        self.simulation.add_object(species_pop_sim_obj)
        return {self.shared_species_store_name: species_pop_sim_obj}

    # TODO(Arthur): not currently used; test after MVP wc_sim done
    def create_access_species_pop(self, lang_submodel):   # pragma: no cover
        """ Create a `LocalSpeciesPopulations` for a submodel and wrap it in an `AccessSpeciesPopulations`

        Args:
            lang_submodel (:obj:`Submodel`): description of a submodel

        Returns:
            :obj:`AccessSpeciesPopulations`: an `AccessSpeciesPopulations` for the `lang_submodel`
        """
        # make LocalSpeciesPopulations & molecular weights
        initial_population = {species_id: self.init_populations[species_id]
                              for species_id in self.private_species[lang_submodel.id]}
        molecular_weights = self.molecular_weights_for_species(self.private_species[lang_submodel.id])

        # DFBA submodels need initial fluxes
        if are_terms_equivalent(lang_submodel.framework, onto['WC:dynamic_flux_balance_analysis']):
            initial_fluxes = {species_id: 0 for species_id in self.private_species[lang_submodel.id]}
        else:
            initial_fluxes = None
        local_species_population = LocalSpeciesPopulation(
            lang_submodel.id.replace('_', '_lsp_'),
            initial_population,
            molecular_weights,
            initial_fluxes=initial_fluxes)

        # make AccessSpeciesPopulations object
        access_species_population = AccessSpeciesPopulations(local_species_population,
                                                             self.species_pop_objs)

        # configure species locations in the access_species_population
        access_species_population.add_species_locations(LOCAL_POP_STORE,
                                                        self.private_species[lang_submodel.id])
        access_species_population.add_species_locations(self.shared_species_store_name,
                                                        self.shared_species)
        return access_species_population
