"""
Event message types, bodies and reply message:
    # TODO(Arthur): include types of objects that send and receive these messages
    CHANGE_POPULATION: species and their population change: dict: species_name -> population_change; no reply message
    GET_POPULATION: list of species whose population is needed; dictionary: species_name -> population
    
    For sequential simulator, store message bodies as a copy of sender's data structure
    # TODO(Arthur): for parallel simulation, use Pickle to serialize and deserialize message bodies
"""

class MessageTypes(object):
    CHANGE_POPULATION = 'CHANGE_POPULATION'
    GET_POPULATION = 'GET_POPULATION'
    GIVE_POPULATION = 'GIVE_POPULATION'

