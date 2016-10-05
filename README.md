# PyCCD - Python Continuous Change Detection
pyccd exists to provide the simplest possible implementation of ccd.

### Getting Started

#### System Requirements
* python3-dev (ubuntu) or python3-devel (centos)
* gfortran
* libopenblas-dev
* liblapack-dev
* graphviz


It's highly recommended to create a virtual environment to perform all
your development and testing.
```bash
user@dev:/home/user/$ mkdir pyccd
user@dev:/home/user/$ cd pyccd
user@dev:/home/user/pyccd$ virtualenv -p python3 .venv
user@dev:/home/user/pyccd$ . .venv/bin/activate
(.venv) user@dev:/home/user/pyccd$
```

All following commands assume an activated virtual environment and pwd as above.  Command prompt is truncated to ```$``` for readability.


#### Get the code
```bash
$ git clone https://github.com/davidvhill/pyccd.git
```

#### Installing
Install test dependencies.
```bash
$ pip install -e .[test]
```

#### Testing & Profiling
```bash
$ python setup.py test
```

Alternatively.
```bash
$ pytest
```

Basic testing with profiling enabled
```bash
$ pytest --profile
```

If an output svg if preferred (useful for performance analysis & optimization):
```bash
$ pytest --profile-svg
```
### Running via command-line
python ./ccd/cli.py

### Developing pyccd

#### app.py

#### cli.py and entry_point scripts
The command line interface is implemented using the click project, which
provides decorators for functions that become command line arguments.

Integration with setup.py entry_point is done via click-plugins, which allow
cli commands to also be designated as entry point scripts.

See ccd.cli.py, setup.py and the click/click-plugin documentation.

* [Click Docs](http://click.pocoo.org/5/)
* [Click On Github](https://github.com/pallets/click)
* [Click on PyPi](https://pypi.python.org/pypi/click)
* [Click-Plugins on Github](https://github.com/click-contrib/click-plugins)
* [Click-Plugins on PyPi](https://pypi.python.org/pypi/click-plugins)


#### logging
Basic Python logging is used in pyccd and is fully configured in app.py.  To use logging in any module:
```python
from ccd import app

logger = app.logger.getLogger(__name__)
logger.info("info level message")
...
```

#### Performance TODO
* optimize data structures (numpy)
* use pypy
* employ @lrucache

### References

### ATBD
1. Obtain minimum number of clear observations
2. Run regression against this set (n)
3. Continue adding observations
4. If three additional consecutive observations falls outside predicted
   values for any band, a change has occurred for all bands
   and a new regression model is started.
5. If next three observations are not outside the range, an outlier has
    has been detected.

* Outliers are flagged and omitted from the regression fitting

* [Test Data](docs/TestData.md)
* [Reference Implementation](https://github.com/USGS-EROS/matlab-ccdc/blob/master/TrendSeasonalFit_v12_30ARDLine.m)
* [Landsat Band Specifications](http://landsat.usgs.gov/band_designations_landsat_satellites.php)
* [Landsat 8 Surface Reflectance Specs](http://landsat.usgs.gov/documents/provisional_lasrc_product_guide.pdf)
* [Landsat 4-7 Surface Reflectance Specs](http://landsat.usgs.gov/documents/cdr_sr_product_guide.pdf)
