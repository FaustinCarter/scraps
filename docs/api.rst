===
API
===

The two main objects that constitue the scraps framework are the :class:`Resonator`
and :class:`ResonatorSweep` objects. Additionally, several helper functions exist
for building or interacting with these objects. There is also a default function
for loading in data from a text file provided, althogh you will likely need to write
your own.

Fitting has been semantically separated, and you will likely want to supply your
own functions for initializing parameters and fitting data, but some default
fit functions are also provided. See the fit functions section of this page.

Plotting is handled by a separate set of plotting functions that take as an argument
either a :class:`Resonator` or :class:`ResonatorSweep` object.

Resonator class
~~~~~~~~~~~~~~~
.. autoclass :: scraps.Resonator

Resonator methods
^^^^^^^^^^^^^^^^^

.load_params()
-----------
.. automethod :: scraps.Resonator.load_params

.torch_params()
---------------
.. automethod :: scraps.Resonator.torch_params

.do_lmfit()
-----------
.. automethod :: scraps.Resonator.do_lmfit

.torch_lmfit()
--------------
.. automethod :: scraps.Resonator.torch_lmfit

.do_emcee()
-----------
.. automethod :: scraps.Resonator.do_emcee

.torch_emcee()
--------------
.. automethod :: scraps.Resonator.torch_emcee

Resonator helper functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

These functions are designed to make it easy to use the :class:`Resonator` class.

process_file()
--------------
.. autofunction :: scraps.process_file

makeResFromData()
-----------------
.. autofunction :: scraps.makeResFromData

makeResList()
-------------
.. autofunction :: scraps.makeResList

indexResList()
--------------
.. autofunction :: scraps.indexResList

ResonatorSweep class
~~~~~~~~~~~~~~~~~~~~
.. autoclass :: scraps.ResonatorSweep

ResonatorSweep methods
^^^^^^^^^^^^^^^^^^^^^^

.do_lmfit()
-----------
.. automethod :: scraps.ResonatorSweep.do_lmfit

.do_emcee()
-----------
.. automethod :: scraps.ResonatorSweep.do_emcee

Plotting tools
~~~~~~~~~~~~~~

plotResListData()
^^^^^^^^^^^^^^^^^
.. autofunction :: scraps.plotResListData

plotResSweepParamsVsTemp()
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction :: scraps.plotResSweepParamsVsTemp

plotResSweepParamsVsPwr()
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction :: scraps.plotResSweepParamsVsPwr

plotResSweep3D()
^^^^^^^^^^^^^^^^
.. autofunction :: scraps.plotResSweep3D

Fit models
~~~~~~~~~~
Each fit model consists of two functions. One that returns a :class:`lmfit.Parameters` object,
and one that takes parameters and data and returns a residual.

I and Q vs frequency
^^^^^^^^^^^^^^^^^^^^
The built-in fit model is called complx_IQ.py and is located in the fitsS21 folder.
It has two functions, one that calculates best guess values for each of the ten fit
parameters, and one that applies those guesses to the data and calculates the residual.

cmplxIQ_params()
----------------
.. autofunction :: scraps.cmplxIQ_params

cmplxIQ_fit()
-------------
.. autofunction :: scraps.cmplxIQ_fit
