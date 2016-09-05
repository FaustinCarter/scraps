from setuptools import setup, find_packages

setup(
    name = 'scraps',
    description = 'SuperConducting Resonator Analysis and Plotting Software.',
    version = 'v0.2.3',
    author = 'Faustin Carter',
    author_email = 'faustin.carter@gmail.com',
    license = 'MIT',
    url = 'http://github.com/faustin315/scraps',
    download_url = 'http://github.com/faustin315/scaps/tarball/v0.1.0',
    packages = ['scraps', 'scraps.fitsS21', 'scraps.fitsSweep'],
    long_description = open('README.rst').read(),
    install_requires = [
        'numpy>=1.5',
        'matplotlib>=1.5',
        'scipy>=0.14',
        'lmfit>=0.9.5',
        'emcee>=2.2.1',
        'pandas>=0.18'
    ],
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Visualization'
    ]

)
