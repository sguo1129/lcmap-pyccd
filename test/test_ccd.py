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


def sinusoid(samples, frequency=1, amplitude=1, seed=42):
    """Produce a sinusoidal wave for testing data"""
    np.random.seed(seed)
    stop = 2 * np.pi * frequency
    xs = np.linspace(0, stop, samples)
    ys = np.array([np.sin(x) * amplitude for x in xs])
    return np.array(list([y + np.random.normal() for y in ys]))


def test_not_enough_observations():
    acquired = acquisition_delta('R15/P16D/2000-01-01')
    reds = sinusoid(15)
    greens = sinusoid(15)
    blues = sinusoid(15)
    observations = np.array([reds, greens, blues])
    fitter_fn = lasso.fitted_model
    models = change.detect(acquired, observations, fitter_fn)
    assert len(models) == 0


def test_enough_observations():
    times = acquisition_delta('R16/P16D/2000-01-01')
    reds = sinusoid(16)
    greens = sinusoid(16)
    blues = sinusoid(16)
    observations = np.array([reds, greens, blues])
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 1, "actual: {}, expected: {}".format(len(models), 1)
    assert len(models[0]) == 3, "actual: {}, expected: {}".format(len(models[0]), 3)


def test_change_windows(n=50, meow_size=16, peek_size=3):
    times = acquisition_delta('R{0}/P16D/2000-01-01'.format(n))
    reds = sinusoid(n)
    greens = sinusoid(n)
    blues = sinusoid(n)
    observations = np.array([reds, greens, blues])
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn, meow_size=meow_size, peek_size=peek_size)
    expected = n - meow_size - peek_size + 2
    assert len(models) == expected, "actual: {}, expected: {}".format(len(models), expected)


def test_two_changes_during_time():
    times = acquisition_delta('R50/P16D/2000-01-01')
    reds = np.hstack((sinusoid(25)+10, sinusoid(25)+50))
    greens = sinusoid(50)
    blues = sinusoid(50)
    observations = np.array([reds, greens, blues])
    fitter_fn = lasso.fitted_model

    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 2, "expected: {0}, actual: {1}".format(2, len(models))


# def test_three_changes_during_time():
#     acquired = acquisition_delta('R90/P16D/2000-01-01')
#     observes = np.hstack((repeated_values(30) + 10,
#                           repeated_values(30) + 50,
#                           repeated_values(30) + 10))
#     assert len(acquired) == len(observes) == 90
