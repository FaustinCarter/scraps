scraps stands for: SuperConducting Resonator Analysis and Plotting Software
=============

Scraps is a package designed to help you analyze lots of data from superconducting
resonators. The basic idea is that you hook up your VNA (or mixer) and measure S21
at several different temperatures and driving powers. Maybe you have a PID and some
decent automation software that take several hundred traces, and now you don't know
what to do with all that data. Enter scraps. Scraps will organize it all and run
fits on it, and make pretty pictures you can show your advisor or funding committee
or spouse (when asked what you do all day upon returning home) or even put into a
journal article.

Currently scraps is designed to handle resonator S21 data at varying temperatures
and input powers. There are plans to extend scraps to handle varying magnetic field
also, as well as noise in addition to S21. If you would like to be a part of that,
get involved by posting a message at the github repo.

Scraps is licensed under the MIT license, so feel free to copy it, play with it,
modify it, etc. under those terms (which are pretty loose!).

Documentation is at: http://scraps.readthedocs.io

If you want to write custom code to load in some type of file that is not a modern Agilent PNA, you'll need a custom process_file function.

Short description of files that aren't code:

- Example1: a JuPyter notebook that will help you get started with the basics.
  Focus is on a single S21 sweep.

- Example2: a slightly more advanced tutorial that focuses on lots of sweeps at
  several temperatures and powers.

- ExampleData.zip: A ton of actual superconducting resonator data from a Nb
  microstrip resonator made at ANL by Dr. Trupti Khaire.

Feedback is always appreciated!
