""" Main bootstrap and configuration module for pyccd.  Any module that
requires configuration or services should import app and obtain the
configuration or service from here.

This pattern is borrowed from flask.
"""
# Logging system
# to use the logging from any module:
# import app
# logger = app.logging.getLogger(__name__)
#
# To alter where log messages go or how they are represented, configure the
# logging system below.
import logging
# iso8601 date format
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

MINIMUM_CLEAR_OBSERVATION_COUNT = 12

# 2 for tri-modal; 2 for bi-modal; 2 for seasonality; 2 for linear
COEFFICIENT_CATEGORIES = {'min': 4, 'mid': 6, 'max': 8}

# number of clear observation / number of coefficients
CLEAR_OBSERVATION_THRESHOLD = 3

QA_CONFIDENCE_FAILURE_THRESHOLD = 0.25

PERMANENT_SNOW_THRESHOLD = 0.75
