from collections import namedtuple

# magnitude, rmse, coefficients, intercept

ccd_result_band = namedtuple("CcdResultBand", ['magnitude', 'rmse',
                                               'coefficients', 'intercept'])

ccd_result = namedtuple("CcdResult", ['start_date', 'end_date',
                                      'red', 'green', 'blue',
                                      'nir', 'swir1', 'swir2',
                                      'thermal',
                                      'category'])


"""
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
"""
