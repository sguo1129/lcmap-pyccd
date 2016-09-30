from collections import namedtuple
import pytest

"""Comprehensive data model of the domain is captured in detections,
observation and observations.  Do not modify these data models unless the
actual domain changes.  Data filtering & transformation should take place
in another module AFTER the functions in change.py have run... as
post-processing steps
"""
# this and all other parameters for the model go into ~/.config/ccd_config.py
# minimum_clear_observation_count = 12
#
# 2 for tri-modal; 2 for bi-modal; 2 for seasonality; 2 for linear
# coefficient_categories = {min:4, mid:6, max:8}
#
# number of clear observation / number of coefficients
# clear_observation_threshold = 3
#
# qa_confidence_failure_threshold = 0.25
# permanent_snow_threshold = 0.75


detections = namedtuple("Detections", ['is_change', 'is_outlier',
                                       'rmse', 'magnitude',
                                       'is_curve_start',
                                       'is_curve_end',
                                       'coefficients',
                                       'category'])

observation = namedtuple('Observation', ['coastal_aerosol', 'red', 'green',
                                         'blue', 'nir', 'swir1',
                                         'swir2', 'panchromatic',
                                         'is_cloud', 'is_clear', 'is_snow',
                                         'is_fill', 'is_water',
                                         'qa_confidence'])

observations = namedtuple('Observations', ['date',
                                           'observation',
                                           'detections'])


def is_change():
    pass


def is_outlier():
    pass


def in_expected_range():
    pass


def regress(observation):
    pass

def error_detector(model):
    """Detect models with RMSE above threshold."""
    return True

def stable(models):
    """Is the RMSE of every model below a threshold?"""
    # return [True for model in models]
    return [error_detector(model) for ix, model in enumerate(models)]

def change_detector(model, peek_values):
    """Detect change outside of tolerance"""
    return True

def accurate(models, moments, spectra):
    """Detect spectral values that do not conform to the model"""
    # return [True for model in models]
    return [change_detector(model, spectra[ix]) for ix, model in enumerate(models)]

def detect(times, observations, fitter_fn, meow_size=16, peek_size=3, keep_all=False):
    """Runs the core ccd algorithm to detect change in the supplied data

    Args:
        observations: a list of acquisition day numbers, and a list
            of spectral values and QA for each day.
        fitter_fn: a function used to fit observation values and
            acquisition dates for each spectra.
        meow_size: minimum expected observation window needed to
            produce a fit.
        peek_size: number of observations to consider when detecting
            a change.
        keep_all: retain all models, even intermediate ones produced
            during initialization and extension.

    Returns:
        Change models for each observation of each spectra.
    """

    # Array index of starting point for initialization.
    meow_ix = 0

    # Result accumulator.
    results = []

    # Only build models as long as sufficient data exists.
    while (meow_ix+meow_size) <= len(times):

        # DEBUGGING? Use `pytest.set_trace()`

        # Initially, there are no models.
        models = None

        # Step 1: INITIALIZATION.
        # The first step is to generate a model that is stable using only
        # the minimum number of observations.
        while (meow_ix+meow_size) <= len(times):

            moments = times[meow_ix:meow_ix + meow_size]
            spectra = observations[:, meow_ix:meow_ix + meow_size]
            models = [fitter_fn(moments,spectrum) for spectrum in spectra]
            results.append(models)

            if not all(stable(models)):
                meow_ix += 1
            else:
                break

        # Determine peek_ix after initialization, this must be done after
        # building the initial model because the window may slide if the
        # initial observations are unstable.
        peek_ix = meow_ix + meow_size

        # Step 2: EXTENSION.
        # The second step is to update a model until observations that do not
        # fit the model are found.
        while (peek_ix+peek_size) < len(times):

            next_moments = times[meow_ix:peek_ix + peek_size]
            next_spectra = observations[:, meow_ix:peek_ix + peek_size]

            if all(accurate(models, next_moments, next_spectra)):
                moments = times[meow_ix:peek_ix + peek_size]
                spectra = observations[:, meow_ix:peek_ix + peek_size]
                models = [fitter_fn(moments, spectrum) for spectrum in spectra]
                results.append(models)
                peek_ix += 1
            else:
                break

        # After exhausting observations that fit the initialized models,
        # reposition the meow_ix to the starting point of the look-ahead
        # so that initialization can begin again.
        meow_ix = peek_ix

    return results
