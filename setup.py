import pip
pip.main(['install', 'git+https://github.com/KarrLab/wc_utils.git#egg=wc_utils'])

from setuptools import setup, find_packages
from wc_utils.util.install import parse_requirements, install_dependencies
import os

# get long description
if os.path.isfile('README.rst'):
    with open('README.rst', 'r') as file:
        long_description = file.read()
else:
    long_description = ''

# get version
with open('wc_sim/VERSION', 'r') as file:
    version = file.read().strip()

# parse dependencies and links from requirements.txt files
with open('requirements.txt', 'r') as file:
    install_requires, dependency_links_install = parse_requirements(file.readlines())
with open('tests/requirements.txt', 'r') as file:
    tests_require, dependency_links_tests = parse_requirements(file.readlines())
dependency_links = list(set(dependency_links_install + dependency_links_tests))

# install non-PyPI dependencies
install_dependencies(dependency_links)

# install package
setup(
    name="wc_sim",
    version=version,
    description="Sequential whole-cell model simulator",
    long_description=long_description,
    url="https://github.com/KarrLab/wc_sim",
    download_url='https://github.com/KarrLab/wc_sim',
    author="Arthur Goldberg",
    author_email="arthur.p.goldberg@mssm.edu",
    license="MIT",
    keywords='whole-cell systems cell molecular biology',
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={
        'wc_sim': [
            'VERSION',
            'core/config/core.default.cfg',
            'core/config/core.schema.cfg',
            'core/config/debug.default.cfg',
            'multialgorithm/config/core.default.cfg',
            'multialgorithm/config/core.schema.cfg',
            'multialgorithm/config/debug.default.cfg',
            'examples/config/debug.default.cfg',            
        ],
    },
    install_requires=install_requires,
    tests_require=tests_require,
    dependency_links=dependency_links,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
