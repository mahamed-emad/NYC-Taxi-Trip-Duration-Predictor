import numpy as np
import pandas as pd
from .functions import *
from .clean_data import *

def add_clms(data):
    dx = abs(data[:, 6] - data[:, 4])
    dy = abs(data[:, 7] - data[:, 5])
    r = dx * 85 + dy * 111

    mn_y, mn_x = -74.25, 40.55
    x_center = (data[:, 6] + data[:, 4] - 2 * mn_x) / 2
    y_center = (data[:, 7] + data[:, 5] - 2 * mn_y) / 2
    
    start = np.asarray((data[:, [4, 6]] - mn_y) * 1e4, dtype=int)
    end = np.asarray((data[:, [5, 7]] - mn_x) * 1e4, dtype=int)
    dx_city = abs(end[:, 1] - start[:, 1])
    dy_city = abs(end[:, 0] - start[:, 0])

    data = np.column_stack((data, r, x_center, y_center, dx_city, dy_city))
    return data

def delete(data):
    data = np.delete(data, [0, 1, 2, 3, 4, 5, 6, 7, 8], axis=1)
    data = np.asarray(data, dtype=float)
    # data : [time, r, center(start, end), num cities in x, num cities in y]
    return data

def remove_outliers(data, mn, mx):
    return data[(data["trip_duration"] >= mn) & (data["trip_duration"] <= mx)]

def clean_clms(data):
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
    data["month"] = data["pickup_datetime"].dt.month
    data["hour"] = data["pickup_datetime"].dt.hour + 1
    data["day_of_week"] = data["pickup_datetime"].dt.dayofweek + 1
    return data