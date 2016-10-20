from setuptools import setup, find_packages
import Sequential_WC_Simulator

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
    install_requires=['cobra', 'matplotlib', 'numpy', 'scipy', 'openpyxl', 
        'future', 'recordtype', 'lxml', 'python-libsbml'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
