"""Multi-algorithm whole-cell simulation utilities. 

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-08-22
:Copyright: 2016, Karr Lab
:License: MIT
"""

def species_compartment_name( specie, compartment ):
    """Provide an identifier for a species in a compartment, formatted  species_id[compartment_it].
    
    Args:
        specie: Species object
        compartment: Compartment object
        
    Returns:
        A unique identifier for a species in a compartment.            
    """
    return "{}[{}]".format( specie.id, compartment.id )
    
def get_species_and_compartment_from_name( species_compartment_name ):
    """The inverse of species_compartment_name().
    
    Args:
        species_compartment_name: string; an identifier for a species in a compartment,
            as produced by species_compartment_name()
        
    Returns: (species, compartment)
    
    Raises:
        ValueError: if species_compartment_name is not of the form "species_id[compartment_id]".
    """
    try:
        (species, rest1 ) = species_compartment_name.split('[')
        (compartment, rest2) = rest1.split(']')
    except ValueError as e:
        raise ValueError( "species_compartment_name must have the form species_id[compartment_id], "
            "but is '{}'".format( species_compartment_name ) )
    return (species, compartment)

