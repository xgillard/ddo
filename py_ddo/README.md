# Py-DDO
PyDDO is a set of python bindings meant to let you use ddo as an engine to perform
your discrete optimization with decision diagrams using python. 

The API is (at the time being at least) much less rich than the rust counter part
still... it works. 

## Limitation
As far as I am concerned, the biggest limitation of using pyddo is that you can
simply not benefit from the parallel capabilities of ddo (the python interpreter
will simply not let you).

At some other places, the interaction between rust and python mean that some 
vectors will have to be instantiated where they could have remained lazy iterators
otherwise... But I suppose this is the kind of performance degradation one is 
willing to accept for the benefit of crafting something with python.

## How do I get this built ? 
In order to use the ddo bindings for python, you will need to have the rust compiler
installed as well as `maturin`. Once that is all setup, all you need to do is:

If you did not already do that, you will want to create a virtual environment where
to compile and install your development version of pyddo.
```
python3 -m venv .env
source .env/bin/activate
```

Once that is done, you launch the compilation of the library as follows:
```
maturin develop
```

You're done, you can use ddo from python
```
python3

import ddo

# yay ! 
```