'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-04-06
:Copyright: 2017-2018, Karr Lab
:License: MIT
'''

from distutils.core import setup, Extension
import os

DIR = os.path.dirname(__file__)

def call_setup():
    module1 = Extension('callbacks',
        sources = [os.path.join(DIR, 'callbacks.c')])

    setup(
        script_name = 'setup.py',
        script_args = ['build'],
        name = 'PackageName',
        version = '0.01',
        description = 'This is a demo callbacks package',
        ext_modules = [module1])