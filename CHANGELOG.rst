Version Next:

* Default colormap is now 'viridis'
* Can omit colorbar if desired
* Added errorbars to plotting
* Proper calculation of errorbars from MCMC
* indexResList can now return a list of all indices matching a given power or
  temperature.
* Documentation update to detail adding custom derived data to ResonatorSweep
  objects.
* Automatic version numbering from VERSION.txt file in setup.py
* Ability to plot multiple different fits on the same axis using plotResSweep3D
* plotResListData now allows kwargs to set upper/lower temp/pwr limits
* Fixed bug where log of S21 magnitude was base e and not base 10; now is 10.

Version 0.2.3:

* Fix importing bug

Version 0.2.0:

* Huge update to plotting. Check it out!
* Bug-fixes and testing for the 2D fitting routines.

Version 0.1.1:

* Bump version for Zenodo

Version 0.1.0:

* Initial version
