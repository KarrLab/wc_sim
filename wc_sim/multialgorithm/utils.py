"""Multi-algorithm whole-cell simulation utilities.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-08-22
:Copyright: 2016, Karr Lab
:License: MIT
"""

# todo: replace with a static method in Species
def get_species_and_compartment_from_name(species_id):
    """Parse a species-compartment name in the form species_type[compartment] into (species_type, compartment)

    The inverse of `Species.gen_id()`.

    Args:
        species_id: string; an identifier for a species in a compartment

    Returns:
        `tuple(str, str)`: (species_type_id, compartment_id)

    Raises:
        ValueError: if species_id is not of the form "species_id[compartment_id]".
    """
    try:
        (species, rest1) = species_id.split('[')
        (compartment, rest2) = rest1.split(']')
    except ValueError as e:
        raise ValueError("species_id must have the form species_id[compartment_id], "
            "but is '{}'".format(species_id))
    return (species, compartment)

