# PyCCD - Python Continuous Change Detection

## Purpose
pyccd exists to provide the simplest possible implementation of ccd.


## System Requirements
* python3-dev (ubuntu) or python3-devel (centos)
* gfortran
* libopenblas-dev
* liblapack-dev
* graphviz


## Getting Started
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


### Get the code
```bash
$ git clone https://github.com/davidvhill/pyccd.git
```

### Developing
Install development dependencies.
```bash
$ pip install -e .[dev]
```

### Testing
Install test dependencies.
```bash
$ pip install -e .[test]
```

Run the tests.
```bash
$ python setup.py test
```

Alternatively.
```bash
$ pytest
```

## Profiling
```bash
$ pytest --profile
```

Or if an output svg if preferred:
```bash
$ pytest --profile-svg
```

## Performance TODO
* optimize data structures (numpy)
* use pypy
* employ @lrucache

## Running via command-line
python ./ccd/cli.py

## Developing


## References

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

### [Test Data](docs/TestData.md)

### [Reference Implementation](https://github.com/USGS-EROS/matlab-ccdc/blob/master/TrendSeasonalFit_v12_30ARDLine.m)

### [Landsat Band Specifications](http://landsat.usgs.gov/band_designations_landsat_satellites.php)

### [Landsat 8 Surface Reflectance Specs](http://landsat.usgs.gov/documents/provisional_lasrc_product_guide.pdf)

### [Landsat 4-7 Surface Reflectance Specs](http://landsat.usgs.gov/documents/cdr_sr_product_guide.pdf)
