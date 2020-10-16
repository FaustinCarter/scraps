import os

from setuptools import find_packages, setup

version_file = open(os.path.join(".", "VERSION.txt"))
version_number = version_file.read().strip()
version_file.close()

setup(
    name="scraps",
    description="SuperConducting Resonator Analysis and Plotting Software.",
    version=version_number,
    author="Faustin Carter",
    author_email="faustin.carter@gmail.com",
    license="MIT",
    url="http://github.com/FaustinCarter/scraps",
    packages=["scraps", "scraps.fitsS21", "scraps.fitsSweep"],
    long_description=open("README.rst").read(),
    install_requires=[
        "numpy",
        "matplotlib>=3",
        "scipy",
        "lmfit>=0.9.5",
        "emcee>=2.2.1",
        "pandas",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
