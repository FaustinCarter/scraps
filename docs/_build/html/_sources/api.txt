===
API
===

The two main objects that constitue the scraps framework are the :class:`Resonator`
and :class:`ResonatorSweep` objects.

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
