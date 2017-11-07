.. scraps documentation master file, created by
   sphinx-quickstart on Wed Aug 31 10:04:27 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scraps: SuperConducting Resonator Analysis and Plotting Software
================================================================

Scraps is a package designed to help you analyze lots of data from superconducting
resonators. The basic idea is that you hook up your VNA (or mixer) and measure S21
at several different temperatures and driving powers. Maybe you have a PID and some
decent automation software that takes several hundred traces, and now you don't know
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

Development of scraps happens over at `github <http://github.com/FaustinCarter/scraps>`_
so that is a great place to post bug reports, requests, kudos, etc.

Contents:
---------

.. toctree::
   :maxdepth: 2

   installation
   Example 1: Introduction <Example1_LoadAndPlot>
   Example 2: Lots of Data <Example2_LotsOfData>
   Example 3: Beautiful Figures <Example3_FiguresForManuscript>
   api
