#!/usr/bin/python

'''Simulates metabolism submodel

:Author: Jonathan Karr, karr@mssm.edu
:Date: 2016-03-24
:Copyright: 2016, Karr Lab
:License: MIT
'''
# TODO(Arthur): IMPORTANT: temporary: refactor and replace

#required libraries
from scipy.constants import Avogadro
from wc_analysis import analysis
from wc_utils.util.rand import RandomStateManager
import numpy as np
import os

#simulation parameters
OUTPUT_DIRECTORY = './plots/'

#plot results
def analyzeResults(model, time, volume, growth, speciesCounts):
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    cellComp = model.get_component_by_id('c', 'compartments')

    totalRna = np.zeros(len(time))
    totalProt = np.zeros(len(time))
    for species in model.species:
        if species.type == 'RNA':
            totalRna += speciesCounts[species.index, cellComp.index, :]
        elif species.type == 'Protein':
            totalProt += speciesCounts[species.index, cellComp.index, :]
	
    '''
    analysis.plot(
        model = model,
        time = time,
        yDatas = {'Volume': volume},
        # COMMENT(Arthur): needs volumetric units, but analysis.plot() doesn't handle them
        # COMMENT(Arthur): one could make 'scale' another argument to plot(), and then
        # use scale and units directly when scale is provided
        fileName = os.path.join(OUTPUT_DIRECTORY, 'Volume.pdf')
        )

    analysis.plot(
        model = model,
        time = time,
        yDatas = {'Growth': growth},
        fileName = os.path.join(OUTPUT_DIRECTORY, 'Growth.pdf')
        )
    '''

    analysis.plot(
        model = model,
        time = time,
        yDatas = {'RNA': totalRna},
        fileName = os.path.join(OUTPUT_DIRECTORY, 'Total RNA.pdf')
        )

    analysis.plot(
        model = model,
        time = time,
        yDatas = {'Protein': totalProt},
        fileName = os.path.join(OUTPUT_DIRECTORY, 'Total protein.pdf')
        )

    analysis.plot(
        model = model,
        time = time,
        volume = volume,
        speciesCounts = speciesCounts,
        units = 'molecules',
        selectedSpeciesCompartments = ['ATP[c]', 'CTP[c]', 'GTP[c]', 'UTP[c]'],
        fileName = os.path.join(OUTPUT_DIRECTORY, 'NTPs.pdf')
        )

    '''
    analysis.plot(
        model = model,
        time = time,
        volume = volume,
        speciesCounts = speciesCounts,
        selectedSpeciesCompartments = ['AMP[c]', 'CMP[c]', 'GMP[c]', 'UMP[c]'],
        units = 'uM',
        fileName = os.path.join(OUTPUT_DIRECTORY, 'NMPs.pdf')
        )

    analysis.plot(
        model = model,
        time = time,
        volume = volume,
        speciesCounts = speciesCounts,
        selectedSpeciesCompartments = ['ALA[c]', 'ARG[c]', 'ASN[c]', 'ASP[c]'],
        units = 'uM',
        fileName = os.path.join(OUTPUT_DIRECTORY, 'Amino acids.pdf')
        )
    '''

    analysis.plot(
        model = model,
        time = time,
        speciesCounts = speciesCounts,
        units = 'molecules',
        selectedSpeciesCompartments = ['RnaPolymerase-Protein[c]', 'Adk-Protein[c]', 'Apt-Protein[c]', 'Cmk-Protein[c]'],
        fileName = os.path.join(OUTPUT_DIRECTORY, 'Proteins.pdf')
        )
