from itertools import cycle, islice
import aniso8601
import datetime
import numpy as np
import ccd.change as change

from ccd.models import lasso


def gen_acquisition_dates(interval):
    """Generate acquisition dates for an ISO8601 interval.

    Args:
        interval: An ISO8610 interval.

    Returns:
        generator producing datetime.date or datetime.datetime
        depending on the resolution of the interval.

    Example:
        gen_acquisition_dates('R10/2015-01-01/P16D')
    """
    dates = aniso8601.parse_repeating_interval(interval)
    return dates


def gen_acquisition_delta(interval):
    """Generate delta in days for an acquisition day since 1970-01-01.

    Args:
        interval: ISO8601 date range

    Returns:
        generator producing delta in days from 1970-01-01

    Example:
        gen_acquisition_delta('R90/P16D/2000-01-01')
    """
    epoch = datetime.datetime.utcfromtimestamp(0).date()
    dates = gen_acquisition_dates(interval)
    yield [(date-epoch).days for date in dates]


def read_csv_sample(path):
    """Load a sample file containing acquisition days and spectral values"""
    return np.genfromtxt('test/resources/sample_1.csv', delimiter=',')


def acquisition_delta(interval):
    """List of delta in days for an interval

    Args:
        interval: ISO8601 date range

    Returns:
        list of deltas in days from 1970-01-01

    Example:
        acquisition_delta('R90/P16D/2000-01-01')
    """
    return list(*gen_acquisition_delta(interval))


def read_csv_sample(path):
    """Load a sample file containing acquisition days and spectral values.

    Args:
        path: location of CSV containing test data

    Returns:
        A 2D numpy array.
    """
    return np.genfromtxt('test/resources/sample_1.csv', delimiter=',')


def repeated_values(samples, seed=42):
    np.random.seed(seed)
    sine = np.array(list(islice(cycle([0, 1, 0, -1]), None, samples)))
    noise = np.array(np.random.random(samples))
    return sine+noise


def test_not_enough_observations():
    acquired = acquisition_delta('R15/P16D/2000-01-01')
    reds = repeated_values(15)
    greens = repeated_values(15)
    blues = repeated_values(15)
    observations = np.array([reds, greens, blues])
    fitter_fn = lasso.fitted_model
    models = change.detect(acquired, observations, fitter_fn)
    assert len(models) == 0


def test_enough_observations():
    times = acquisition_delta('R16/P16D/2000-01-01')
    reds = repeated_values(16)
    greens = repeated_values(16)
    blues = repeated_values(16)
    observations = np.array([reds, greens, blues])
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 1, "actual: {}, expected: {}".format(len(models), 1)
    assert len(models[0]) == 3, "actual: {}, expected: {}".format(len(models[0]), 3)


def test_change_windows(n=50, meow_size=16, peek_size=3):
    times = acquisition_delta('R{0}/P16D/2000-01-01'.format(n))
    reds = repeated_values(n)
    greens = repeated_values(n)
    blues = repeated_values(n)
    observations = np.array([reds, greens, blues])
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn, meow_size=meow_size, peek_size=peek_size)
    expected = n - meow_size - peek_size + 2
    assert len(models) == expected, "actual: {}, expected: {}".format(len(models), expected)


def test_two_changes_during_time():
    times = acquisition_delta('R50/P16D/2000-01-01')

    # The red band has two distinct segments, but the green and blue bands
    # are consistent.
    reds = np.hstack((repeated_values(25)+10, repeated_values(25)+50))
    greens = repeated_values(50)
    blues = repeated_values(50)
    observations = np.array([reds, greens, blues])

    fitter_fn = lasso.fitted_model

    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 1, "expected: {0}, actual: {1}".format(1, len(models))


# def test_three_changes_during_time():
#     acquired = acquisition_delta('R90/P16D/2000-01-01')
#     observes = np.hstack((repeated_values(30) + 10,
#                           repeated_values(30) + 50,
#                           repeated_values(30) + 10))
#     assert len(acquired) == len(observes) == 90
