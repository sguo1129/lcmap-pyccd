from collections import namedtuple
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


def change_detector(model, peek_values):
    """Detect change outside of tolerance"""
    return False

def unchanged(models, peek_values, change_detect_fn):
    return not any([change_detect_fn(model, peek_values[ix]) for ix, model in enumerate(models)])


def detect(times, observations, fitter_fn, change_detect_fn, meow_size=16, peek_size=3):
    """Runs the core ccd algorithm to detect change in the supplied data

    Args:
        observations: a list of acquisition day numbers, and a list
            of spectral values and QA for each day.
        fitter: a function used to fit observation values and
            acquisition dates for each spectra.
        changer: a function used to detect change; expects a model
            and an observation.
        meow_size: minimum expected observation window needed to
            produce a fit.
        peek_size: number of observations to consider when detecting
            a change.

    Returns:
        Change models for each observation of each spectra.
    """

    # Array index of the model start. Obviously this starts at
    # zero, but as changes are detected, this will be updated
    # with a value of the array index, the new model's start.
    start_ix = 0

    # Result accumulator. Each observation of each spectra has an
    # updated model.
    results = []

    # Array index of where to begin peeking at new observations.
    # Initially, this is the minimum expected observation window,
    # (meow_size).
    peek_ix = start_ix + meow_size

    # There are more observations to consider
    while (start_ix + meow_size) <= len(times):

        # Build a model for each spectra
        moments = times[start_ix:meow_size]
        spectra = observations[:,start_ix:meow_size]

        # If there are enough observations, fit the model.
        if len(moments) >= meow_size:
            models = [fitter_fn(moments,spectrum) for spectrum in spectra]
            results.append(models)
        else:
            break

        # Update models while things appear stable (i.e. none of the
        # observation spectra peek-windows exhibit change).
        while (peek_ix+peek_size) <= len(times):
            # If the models for all spectra appear unchanged, then run the
            # fitting function using a window that extends into the peeked
            # observations.
            peek_values = observations[:, peek_ix:peek_ix+peek_size]
            if unchanged(models, peek_values, change_detector):
                # TODO (jmorton) determine how to handle outliers.
                moments = times[start_ix:(peek_ix+peek_size)]
                spectra = observations[:, start_ix:(peek_ix+peek_size)]
                updates = [fitter_fn(moments, spectrum) for spectrum in spectra]
                results.append(updates)
                peek_ix += 1
                print("continue existing time segment: {} {}".format(start_ix,peek_ix+peek_size))
                continue

            # Otherwise, if any spectra's peeked values appear to change, begin
            # a new segment starting at the peeked index. The
            else:
                start_ix = peek_ix
                peek_ix = start_ix + meow_size
                print("begin new time segment: {} {}".format(start_ix,peek_ix+peek_size))
                break

        break

    return results
