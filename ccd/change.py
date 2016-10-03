from collections import namedtuple
import numpy as np
from ccd.models import lasso

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


def rmse(models, moments, spectra):
    """Calculate RMSE for all models; used to determine if models are stable.

    Args:
        models: fitted models, used to predict values
        moments: ordinal day numbers
        spectra: list of spectra corresponding to models

    Returns:
        list: RMSE for each model.
    """
    errors = []
    for model, observed in zip(models, spectra):
        matrix = lasso.coefficient_matrix(moments)
        predictions = model.predict(matrix)
        # TODO (jmorton): VERIFY CORRECTNESS
        error = np.linalg.norm(predictions - observed) / np.sqrt(len(predictions))
        errors.append(error)
    return errors


def stable(models, moments, spectra, threshold=2.0):
    """Determine if all models RMSE are below threshold.

    Args:
        models: fitted models, used to predict values.
        moments: ordinal day numbers.
        spectra: list of spectrum corresponding to models.

    Returns:
        bool: True, if all models RMSE is below threshold, False otherwise.
    """
    errors = rmse(models, moments, spectra)
    return all([e < 2.0 for e in errors])


def magnitudes(models, moments, spectra):
    """Calculate change magnitudes for each model and spectra.

    Magnitude is the 2-norm of the difference between predicted
    and observed values.

    Args:
        models: fitted models, used to predict values.
        moments: ordinal day numbers.
        spectra: list of spectrum corresponding to models
        threshold: tolerance between detected values and
            predicted ones.

    Returns:
        list: magnitude of change for each model.
    """
    magnitudes = []
    for model, observed in zip(models, spectra):
        matrix = lasso.coefficient_matrix(moments)
        predicted = model.predict(matrix)
        # TODO (jmorton): VERIFY CORRECTNESS
        # This approach matches what is done if 2-norm (largest sing. value)
        magnitude = np.linalg.norm((predicted-observed), ord=2)
        magnitudes.append(magnitude)
    return magnitudes


def accurate(models, moments, spectra, threshold=0.99):
    """Are observed spectral values within the predicted values' threshold.

    Args:
        models: fitted models; used to predict values
        moments: ordinal day numbers
        spectra: list of spectrum corresponding to models
        threshold: tolerance between detected values and predicted ones

    Returns:
        bool: True if each model's predicted and observed values are
            below the threshold, False otherwise.
    """
    ms = magnitudes(models, moments, spectra)
    return all([m < threshold for m in ms])


def find_time_index(times, meow_ix, meow_size, day_delta = 365):
    """Find index in times at least one year from time at meow_ix.
    Args:
        times: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        meow_ix: index into times, used to get day number for comparing
            times for
        meow_size: relative number of observations after meow_ix to
            begin searching for a time index
        day_delta: number of days difference between meow_ix and
            time index
    Returns:
        integer: array index of time at least one year from meow_ix,
            or None if it can't be found.
    """

    # If the last time is less than a year, then iterating through
    # times to find an index is futile.
    if times[-1] - times[meow_ix] < day_delta:
        return None

    # Array index is zero based, so the the end index needs to be
    # subtracted by one.
    end_ix = meow_ix + meow_size - 1

    # This seems pretty naive, if you can think of something more
    # performant and elegant, have at it!
    while end_ix < len(times):
        if times[end_ix] - times[meow_ix] >= 365:
            return end_ix
        else: # try again!
            end_ix += 1


def detect(times, observations, fitter_fn, meow_size=16, peek_size=3, keep_all=False):
    """Runs the core change detection algorithm.

    The algorithm assumes all pre-processing has been performed on
    observations.

    Args:
        times: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: values for one or more spectra corresponding
            to each time.
        fitter_fn: a function used to fit observation values and
            acquisition dates for each spectra.
        meow_size: minimum expected observation window needed to
            produce a fit.
        peek_size: number of observations to consider when detecting
            a change.
        keep_all: retain all models, even intermediate ones produced
            during initialization and extension.

    Returns:
        list: Change models for each observation of each spectra.
    """

    # Accumulator for models. This is a list of lists; each top-level list
    # corresponds to a particular spectra.
    results = []

    # Alternate accumulator for models when `keep_all` is True. Having two
    # separate collections makes the code easier to understand.
    keeper = []

    # The starting point for initialization. Used to as reference point for
    # taking a range of times and spectral values.
    meow_ix = 0

    # Only build models as long as sufficient data exists. The observation
    # window starts at meow_ix and is fixed until the change model no longer
    # fits new observations, i.e. a change is detected.
    while (meow_ix+meow_size-1) <= len(times):

        # Step 1: INITIALIZATION.
        # The first step is to generate a model that is stable using only
        # the minimum number of observations.
        while (meow_ix+meow_size-1) <= len(times):

            # Stretch observation window until it includes full year.
            end_ix = find_time_index(times, meow_ix, meow_size)

            # If an end index could not be found, then there aren't enough
            # observations to continue producing models.
            if end_ix is None:
                return results

            moments = times[meow_ix:end_ix]
            spectra = observations[:, meow_ix:end_ix]
            models = [fitter_fn(moments, spectrum) for spectrum in spectra]

            if keep_all:
                keeper.append(models)

            # If a model is not stable, then it is possible that a disturbance
            # exists somewhere in the observation window. The window shifts
            # forward in time, and begins initialization again.
            if not stable(models, moments, spectra):
                meow_ix += 1
            else:
                break

        # Determine peek_ix after initialization, this must be done after
        # building the initial model because the window may slide if the
        # initial observations are unstable.
        peek_ix = meow_ix + end_ix

        # Step 2: EXTENSION.
        # The second step is to update a model until observations that do not
        # fit the model are found.
        while (peek_ix+peek_size) <= len(times):
            print("Peeking from {}..{}".format(peek_ix, peek_ix+peek_size))
            next_moments = times[meow_ix:peek_ix + peek_size]
            next_spectra = observations[:, meow_ix:peek_ix + peek_size]

            if accurate(models, next_moments, next_spectra):
                print("Extending an accurate model.")
                moments = times[meow_ix:peek_ix + peek_size]
                spectra = observations[:, meow_ix:peek_ix + peek_size]
                models = [fitter_fn(moments, spectrum) for spectrum in spectra]
                peek_ix += 1
                if keep_all:
                    keeper.append(models)
            else:
                print("{} peeked values do not fit, breaking extension.".format(peek_size))
                break

        # After exhausting observations that fit the initialized models,
        # reposition the meow_ix to the starting point of the look-ahead
        # so that initialization can begin again.
        print("Accumulating model.")
        results.append(models)
        meow_ix = peek_ix

    if keep_all:
        return keeper
    else:
        return results
