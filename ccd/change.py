from collections import namedtuple
import numpy as np
from ccd.models import lasso


def rmse(models, moments, spectra):
    """Is the RMSE of every model below a threshold???"""
    errors = []
    for model, observed in zip(models, spectra):
        matrix = lasso.coefficient_matrix(moments)
        predictions = model.predict(matrix)
        error = np.linalg.norm(predictions - observed) / np.sqrt(len(predictions))
        errors.append(error)
    return errors


def stable(models, moments, spectra):
    """"""
    errors = rmse(models, moments, spectra)
    print("rmse: {0}".format(errors))
    return not any([e > 2.0 for e in errors])


def magnitudes(models, moments, spectra):
    """Calculate change magnitudes for each model and spectra"""
    magnitudes = []
    for model, observed in zip(models, spectra):
        matrix = lasso.coefficient_matrix(moments)
        predicted = model.predict(matrix)
        magnitude = np.linalg.norm((predicted-observed), ord=2)
        magnitudes.append(magnitude)
    return magnitudes


def accurate(models, moments, spectra, threshold=0.99):
    """Detect spectral values that do not conform to the model"""
    ms = magnitudes(models, moments, spectra)
    print("mags: {0}".format(ms))
    return not any([m > threshold for m in ms])


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

    # Accumulator for models. This is a list of lists; each top-level list
    # corresponds to a particular spectra.
    results = []

    # The starting point for initialization. Used to as reference point for
    # taking a range of times and spectral values.
    meow_ix = 0

    # Only build models as long as sufficient data exists. The observation
    # window starts at meow_ix and is fixed until the change model no longer
    # fits new observations, i.e. a change is detected.
    while (meow_ix+meow_size) <= len(times):

        # Step 1: INITIALIZATION.
        # The first step is to generate a model that is stable using only
        # the minimum number of observations.
        # TOD0 (jmorton): determine how time factors into window; is it
        #      necessary to have a minimm number of observations AND
        #      an entire year of data?
        while (meow_ix+meow_size) <= len(times):

            moments = times[meow_ix:meow_ix + meow_size]
            spectra = observations[:, meow_ix:meow_ix + meow_size]
            models = [fitter_fn(moments, spectrum) for spectrum in spectra]

            if not stable(models, moments, spectra):
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
        while (peek_ix+peek_size) <= len(times):

            next_moments = times[meow_ix:peek_ix + peek_size]
            next_spectra = observations[:, meow_ix:peek_ix + peek_size]

            if accurate(models, next_moments, next_spectra):
                moments = times[meow_ix:peek_ix + peek_size]
                spectra = observations[:, meow_ix:peek_ix + peek_size]
                models = [fitter_fn(moments, spectrum) for spectrum in spectra]
                peek_ix += 1
            else:
                break

        # After exhausting observations that fit the initialized models,
        # reposition the meow_ix to the starting point of the look-ahead
        # so that initialization can begin again.
        results.append(models)
        meow_ix = peek_ix

    return results
