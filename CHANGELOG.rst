Version Next:

* Fixed a bug in the ResonatorSweep smart-indexing logic by completely reworking the algorithm.

Version 0.4.2:

* Changed how tick labels are rotated to fix strange rounding errors on tick labels and problems
  with tick labels not respecting interactive plot resizing.
* Added a `x_slice` keyword argument to the `plot_tools.plotResListData` function. This adds the
  ability to only plot a portion of data. Currently does not check whether all resonators in list
  have the same length data, so be aware of potential for an exception.

Version 0.4.1:

* More bug fixes that were introduced with the changes to fit storage in 0.4.0
* Added kwarg to `process_file` function to allow stripping out JSON metadata stored in first line

Version 0.4.0:

* Change the way fits are stored in the resonator object. *This slightly breaks backwards compatibility.*
  Results that used to live at `Resonator.lmfit_result` or `Resonator.emcee_result` now live at
  `Resonator.lmfit_result['default']['result']` and `Resonator.emcee_result['default']['result']`. This allows the user to store
  multiple fit results for the same resonator by specifying a label for the fit (as opposed to using the
  `'default'` label).
* Update dependency on matplotlib to be >= 2.0
* Fix minus-sign error in cmplxIQ fit function and params guessing (each had one error, was cancelling)
* Add a burn_flatchain method to each resonator to allow burning off some samples from the mcmc analysis
* Add a function to ESL_tools to read in binary fits files coming out of ESL
* Add a mask option to process_file to allow for masking data via a slice object
* Change under the hood to how fitting works for Resonator objects (and eventually to ResonatorSweep data as well).
  Now instead of being forced into fitting only I/Q data, fit functions take a Resonator object as an argument.
  This gives the fit function author access to all of the attributes and data stored in the object.
* Update the complxIQ fit function to calculate not just the model, but the baseline, or even the model without the baseline.
* Allow the user to pass a frequency vector to cmplxIQ to specify which frequency points to calculate the model at.
* Fix a small bug in plot kwargs checking introduced during update to Python 3.

Version 0.3.1:

* Fix README formatting for PyPI

Version 0.3.0:

* Fixed a bug in the plotting routines that was drawing incorrect x-axis labels
* Added a keyword arguments passthrough to numpy.loadtxt for tuning how process_file loads a file
* Huge plotting update! Now can plot data from multiple ResonatorSweep objects on the same axis
* Compatible with Python 2 and 3
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
* Fixed bug where log of S21 magnitude was base e and not base 10; now is 10
* Added option to fit a quadratic function to the phase baseline
* Added models for qi and f0 based on full Mattis-Bardeen calculations from barmat
* Added some helper utilities for viewing resList data structure

Version 0.2.3:

* Fix importing bug

Version 0.2.0:

* Huge update to plotting. Check it out!
* Bug-fixes and testing for the 2D fitting routines.

Version 0.1.1:

* Bump version for Zenodo

Version 0.1.0:

* Initial version
