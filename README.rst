scraps stands for: SuperConducting Resonator Analysis and Plotting Software

scraps is a project designed to make processing superconducting resonator data as painless as possible.

Documentation is at: http://scraps.readthedocs.io

If you want to write custom code to load in some type of file that is not a modern Agilent PNA, you'll need a custom process_file function.

Short description of files:

resonator.py - Defines the Resonator object
resonator_sweep.py - Defines the ResonatorSweep object, which is designed to help you when you have so much data.
plot_tools.py - Helps you make beautiful plots from the data in your Resonator and ResonatorSweep objects
process_file.py - Get data into pyRes
fitsS21 - directory containing fit functions for resonator data
fitsSweep - directory containing fit functions for more complicated things
resMCMC.py - experimental code for integrating emcee with scraps. No longer used now that lmfit has this implemented directly.
Example1 - Start here!
Example2 - Then do this one!
ExampleData.zip - A ton of actual superconducting resonator data from a Nb microstrip resonator made at ANL

Feedback is always appreciated.

-FC
