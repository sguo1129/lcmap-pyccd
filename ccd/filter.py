"""Filters for pre-processing change model inputs.
"""


def ratio_clear(qa):
    """Calculate ratio of clear to non-clear pixels; exclude, fill data."""
    # TODO (jmorton) Verify; does the ratio exclude fill data?
    fill = 255
    clear_count = qa[(qa < 2)].shape[0]
    total_count = qa[(qa < fill)].shape[0]
    return clear_count / total_count


def enough_clear(qa, threshold):
    """Determine if clear observations exceed threshold.

    Useful when selecting mathematical model for detection."""
    return ratio_clear(qa) >= threshold


def ratio_snow(qa):
    """Calculate ratio of snow to clear pixels; exclude fill and non-clear data."""
    # TODO (jmorton) Verify; does the ratio exclude fill?
    # TODO (jmorton) Do we need to add 0.01 to the result like the Matlab version?
    snow = 4
    snowy_count = qa[(qa == snow)].shape[0]
    clear_count = qa[(qa < 2)].shape[0]
    return snowy_count / (total_count+snowy_count)


def enough_snow(qa, threshold):
    """Determine if snow observations exceed threshold.

    Useful when selecting detection algorithm."""
    return ratio_snow(qa) >= threshold


def unsaturated_index(observations):
    """Produce bool index for observations that are unsaturated (values between 0..10,000)

    Useful for efficiently filtering nd arrays."""
    # TODO (jmorton) Is there a more concise way to provide this function
    #      without being explicit about the expected dimensionality of the
    #      observations?
    unsaturated = ((0 < xs[:,1]) & (xs[:,1] < 10000) &
                   (0 < xs[:,2]) & (xs[:,2] < 10000) &
                   (0 < xs[:,3]) & (xs[:,3] < 10000) &
                   (0 < xs[:,4]) & (xs[:,4] < 10000) &
                   (0 < xs[:,5]) & (xs[:,5] < 10000) &
                   (0 < xs[:,6]) & (xs[:,6] < 10000))
    return unsaturated


def temperature_index(observations):
    """ 0 to 10000 for all bands but thermal.  Thermal is -93.2C to 70.7C
    179.95K -- 343.85K """
    # Temperature check.
    # TODO (jmorton) These are parameters and should not be hard-coded.
    min_kelvin, max_kelvin = 179.95, 343.85
    temperature_index = ((min_kelvin <= observations[:,7])&
                         (observations[:,7] <= max_kelvin))
    return temperature_index


def categorize(qa):
    """ determine the category to use for detecting change """
    """
    IF clear_pct IS LESS THAN CLEAR_OBSERVATION_THRESHOLD
    THEN

        IF permanent_snow_pct IS GREATER THAN PERMANENT_SNOW_THRESHOLD
        THEN
            IF ENOUGH snow pixels
            THEN
                DO snow based change detection
            ELSE
                BAIL

    ELSE
        DO NORMAL CHANGE DETECTION


    """


def preprocess(matrix):
    pass
