# PyNeurActiv

Python module on neuronal activity: analysis tools and theoretical models.


## Installation

This Python package requires:

- a working installation of Python >= 2.7
- ``numpy`` 
- ``scipy``
- ``matplotlib``

Recommended additional packages are:

- [nngt](https://github.com/Silmathoron/NNGT) for some functionalities
- [pandas](http://pandas.pydata.org/) for raster analysis


Open a terminal in the folder of your choice, then type

    git clone https://github.com/SENeC-Initiative/PyNeurActiv.git
    cd PyNeurActiv
    export PYTHONPATH=$(pwd):$PYTHONPATH

otherwise you can also install it globally with ``pip``

    git clone https://github.com/SENeC-Initiative/PyNeurActiv.git
    cd PyNeurActiv
    sudo pip install -e .

The above commands will temporarily set your `$PYTHONPATH`.
Add the appropriate path in your `.bashrc` to set the package permanently.

Package can then be imported in python using ``import PyNeurActiv``
or for short ``import PyNeurActiv as pna``.
