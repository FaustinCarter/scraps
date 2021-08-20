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
--------------
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
The ResonatorSweep class inherits python's ``dict`` class. Each key of the dict
corresponds to a ``pandas.DataFrame`` of data. Because of this, you can add your
own custom derived data sets by simply assigning them to a dict entry. As an
example::

  #Start with a ResonatorSweep object called resSweep that contains a
  #key called 'f0', which is the resonant frequency of many different
  #resonators. Maybe what you want to plot though, is the reduced
  #frequency df = (f-fr)/fr where fr is some reference frequency

  #Choose a reference frequency from the existing data
  fr = resSweep['f0'].iloc[0,0]

  #Pick a descriptive key to describe the data
  key = 'df_over_fr'

  #Calculate the derived quantity and add it to the ResonatorSweep
  resSweep[key] = (resSweep['f0']-fr)/fr

  #And now you can plot it by just passing that key when you use the
  #plotting tools. If you want errorbars, you'll need to calculate those
  #for the derived quantity as well, and store them in: key + '_sigma'.

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

plotResSweepParamsVsX()
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction :: scraps.plotResSweepParamsVsX

plotResSweepParamsVsTemp()
^^^^^^^^^^^^^^^^^^^^^^^^^^
Note: this has been deprecated in favor of plotResSweepParamsVsX
.. autofunction :: scraps.plotResSweepParamsVsTemp

plotResSweepParamsVsPwr()
^^^^^^^^^^^^^^^^^^^^^^^^^
Note: this has been deprecated in favor of plotResSweepParamsVsX
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
The built-in fit model is called hanger_resonator.py and is located in the fitsS21 folder.
It has two functions, one that calculates best guess values for each of the ten fit
parameters, and one that applies those guesses to the data and calculates the residual.

*Note:* This module used to be called cmplxIQ.py. The hanger functions are the same, just renamed.

hanger_params()
----------------
.. autofunction :: scraps.hanger_params

hanger_fit()
-------------
.. autofunction :: scraps.hanger_fit

Two-level system (TLS) and Mattis-Bardeen effect (BMD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This model is for fitting frequency shifts and internal quality factors as functions
of temperature and input power. It is a very simple model, employs a lot of
simplifying assumptions, and should be regarded with extreme skepticism. However,
it qualitatively describes the dominant behavior of most resonators and so is
useful as an example.

There is no accompanying parameter-generation function. See Example 3 for usage.

qi_tlsAndMBT()
----------------------------
.. autofunction :: scraps.fitsSweep.qi_tlsAndMBT

f0_tlsAndMBT()
-----------------------------
.. autofunction :: scraps.fitsSweep.f0_tlsAndMBT
