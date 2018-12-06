import pandas as pd
import numpy as np

from pandas.core.nanops import nanmean


def get_seasonality(row, window_size):
    """
    Returns seasonality component of a time series
    :param row: pandas series containing the time series to extract seasonality from
    :param window_size: length of moving window
    :return: pandas series of seasonality component
    """
    averages = np.array([nanmean(row[i::window_size], axis=0) for i in range (window_size)])
    averages -= np.mean(averages, axis=0)
    season = np.tile(averages.T, len(row)//window_size+1).T[:len(row)]

    return pd.Series(season)


def get_trend(row, window_size, center=True):
    """
    Returns trend component of a time series
    :param row: pandas series containing the time series to extract seasonality from
    :param window_size: length of moving window
    :param center:
    :return:
    """
    trend = row.rolling(window_size, center=center).mean()

    return trend