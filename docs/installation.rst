============
Installation
============

Required packages
~~~~~~~~~~~~~~~~~

Scraps is written in Python 3.6 (compatible with 2.7) and requires the following packages:

* numpy
* pandas
* scipy
* lmfit
* emcee (optional)
* matplotlib


Downloading and installing
~~~~~~~~~~~~~~~~~~~~~~~~~~

The latest stable version of scraps is hosted at `PyPi
<http://pypi.python.org/pypi/scraps/>`_.

If you like pip, you can install with::

  pip install scraps

Or, alternatively, download and extract the tarball from the above link and then
install from source with::

  cd directory_you_unpacked_scraps_into
  python setup.py install


Development
~~~~~~~~~~~

Development happens at `github <http://github.com/FaustinCarter/scraps>`_. You can
clone the repo with::

  git clone http://github.com/FaustinCarter/scraps

And you can install it in developer mode with pip::

  pip install -e /directory/where/you/cloned/scraps

or from source::

  cd /directory/where/you/cloned/scraps
  python setup.py develop

For the bleeding edge version checkout the develop branch::

  cd /directory/where/you/cloned/scraps
  git checkout develop

If you like this project and want to contribute, send a message over at the
github repo and you can be added as a collaborator!
