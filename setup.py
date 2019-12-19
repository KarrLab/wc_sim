import setuptools
try:
    import pkg_utils
except ImportError:
    import pip._internal.main
    pip._internal.main.main(['install', 'pkg_utils'])
    import pkg_utils
import os

name = 'wc_sim'
dirname = os.path.dirname(__file__)

# get package metadata
md = pkg_utils.get_package_metadata(dirname, name)

# install package
setuptools.setup(
    name=name,
    version=md.version,
    description="Multialgorithmic whole-cell model simulator",
    long_description=md.long_description,
    url="https://github.com/KarrLab/" + name,
    download_url='https://github.com/KarrLab/' + name,
    author="Arthur Goldberg",
    author_email="arthur.p.goldberg@mssm.edu",
    license="MIT",
    keywords='whole-cell systems cell molecular biology',
    packages=setuptools.find_packages(exclude=['tests', 'tests.*']),
    package_data={
        name: [
            'VERSION',
            'core/config/core.default.cfg',
            'core/config/core.schema.cfg',
            'core/config/debug.default.cfg',
            'config/core.default.cfg',
            'config/core.schema.cfg',
            'config/debug.default.cfg',
            'examples/config/debug.default.cfg',
        ],
    },
    entry_points={
        'console_scripts': [
            'wc-sim = wc_sim.__main__:main',
        ],
    },
    install_requires=md.install_requires,
    extras_require=md.extras_require,
    tests_require=md.tests_require,
    dependency_links=md.dependency_links,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
