from setuptools import setup, find_packages
from wc_utils.util.installation import parse_requirements, install_dependencies_from_links
import Sequential_WC_Simulator

# parse dependencies and links from requirements.txt files
with open('requirements.txt', 'r') as file:
    install_requires, dependency_links_install = parse_requirements(file.readlines())
with open('tests/requirements.txt', 'r') as file:
    tests_require, dependency_links_tests = parse_requirements(file.readlines())
dependency_links = list(set(dependency_links_install + dependency_links_tests))

# install non-PyPI dependencies
install_dependencies_from_links(dependency_links)

# install package
setup(
    name="Sequential_WC_Simulator",
    version=Sequential_WC_Simulator.__version__,
    description="Sequential whole-cell model simulator",
    url="https://github.com/KarrLab/Sequential_WC_Simulator",
    download_url='https://github.com/KarrLab/Sequential_WC_Simulator/tarball/{}'.format(Sequential_WC_Simulator.__version__),
    author="Arthur Goldberg",
    author_email="arthur.p.goldberg@mssm.edu",
    license="MIT",
    keywords='whole-cell systems cell molecular biology',
    packages=find_packages(exclude=['tests', 'tests.*']),
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
