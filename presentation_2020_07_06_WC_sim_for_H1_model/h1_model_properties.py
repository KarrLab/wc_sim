""" Obtain H1 model properties for presentation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-07-04
:Copyright: 2020, Karr Lab
:License: MIT
"""

from collections import defaultdict
from pprint import pprint
import itertools
import networkx
import os
import pandas

from de_sim.simulation_config import SimulationConfig
from wc_sim.dynamic_components import CacheManager
from wc_sim.sim_config import WCSimulationConfig
import obj_tables
import wc_lang
import wc_lang.io

TEST_DEPENDENCIES_MODEL = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures',
                                       'test_dependencies.xlsx')
MOCK_H1_HESC_MODEL = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures',
                                  'mock_h1_hesc_model_deterministic.xlsx')
H1_MODEL_SCALED_DOWN = os.path.join(os.path.dirname(__file__), '..', '..', 'h1_hesc', 'h1_hesc',
                                    'scaled_down_model', 'model_core.xlsx')

def make_dynamic_model(model_filename):
    # read and initialize a model
    model = wc_lang.io.Reader().run(model_filename, validate=False)[wc_lang.Model][0]
    de_simulation_config = SimulationConfig(max_time=10)
    wc_sim_config = WCSimulationConfig(de_simulation_config, ode_time_step=2)
    multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
    _, dynamic_model = multialgorithm_simulation.build_simulation()
    return model, dynamic_model

def obtain_expression_usage(model):
    """ Obtain the use of expressions by rate laws in a WC Lang model

    Adapted from DynamicModel.obtain_dependencies.

    Args:
        model (:obj:`Model`): the whole-cell model

    Returns:
        :obj:`dict`: the quantitative use of expressions by rate laws in a WC Lang model
    """
    used_model_types = set((wc_lang.Function,
                            wc_lang.Observable,
                            wc_lang.RateLaw,
                            wc_lang.Species,
                            wc_lang.StopCondition,
                            wc_lang.Compartment))

    model_entities = itertools.chain(model.functions,
                                     model.observables,
                                     model.rate_laws,
                                     model.stop_conditions)

    # 1) make digraph of dependencies among model instances
    dependencies = networkx.DiGraph()
    for dependent_model_entity in model_entities:

        dependent_model_entity_expr = dependent_model_entity.expression

        # get all instances of types in used_model_types used by dependent_model_entity
        used_models = []
        for attr_name, attr in dependent_model_entity_expr.Meta.attributes.items():
            if isinstance(attr, obj_tables.RelatedAttribute):
                if attr.related_class in used_model_types:
                    used_models.extend(getattr(dependent_model_entity_expr, attr_name))

        # add edges from dependent_model_entity to the model_type entities on which it depends
        for used_model in used_models:
            dependencies.add_edge(dependent_model_entity, used_model)

    # 2) add all rate laws
    for rate_law in model.rate_laws:
        dependencies.add_node(rate_law)

    # SKIP 3) a compartment in an expression is a special case that computes the compartment's mass
    # add dependencies between each compartment used in an expression and all the species in the compartment

    # 4) find the expressions that each rate law uses
    n = 0
    next = 1
    rate_law_expr_use = {}
    for rate_law in model.rate_laws:
        n += 1
        if n == next:
            print(f"processing rate law {n}")
            next *= 2
        rate_law_expr_use[rate_law.id] = defaultdict(int)
        for expression in dependencies.nodes:
            if isinstance(expression, wc_lang.RateLaw):
                break
            for path in networkx.all_simple_paths(dependencies, rate_law, expression):
                rate_law_expr_use[rate_law.id][expression] += 1

    return rate_law_expr_use

def get_h1_model_properties(model_file):
    ### static data ###
    # read the model
    h1_model = wc_lang.io.Reader().run(model_file, validate=False)[wc_lang.Model][0]
    # get comprehensive model entitiy counts
    interesting_entities = ('submodels', 'compartments', 'species_types', 'species', 'observables', 'functions',
                            'reactions', 'rate_laws', 'stop_conditions', 'parameters')
    comprehensive_entitiy_counts = {}
    for interesting_entity in interesting_entities:
        entities = getattr(h1_model, interesting_entity)
        interesting_entity_name = interesting_entity
        if entities:
            interesting_entity_name = entities[0].__class__.__name__
        count = len(entities)
        comprehensive_entitiy_counts[interesting_entity_name] = count
    print(f"H1 model component\tCount")
    for interesting_entity_name, count in comprehensive_entitiy_counts.items():
        print(f"{interesting_entity_name}\t{count}")
    print()
    # todo: get model entitiy counts, by submodel
    # get distributions of expression use by rate laws
    expression_usage = pandas.DataFrame.from_dict(obtain_expression_usage(h1_model))
    usage_filename = os.path.splitext(os.path.basename(model_file))[0] + '_expression_usage.tsv'
    expression_usage.to_csv(usage_filename, sep='\t')
    print('wrote', usage_filename)

get_h1_model_properties(MOCK_H1_HESC_MODEL)
get_h1_model_properties(H1_MODEL_SCALED_DOWN)
