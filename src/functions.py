import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def to_np(data):
    return data.to_numpy()

def to_pd(data):
    return pd.DataFrame(data)

def log(clm):
    return np.log1p(clm)

def exp(clm):
    return np.expm1(clm)

def split_data(data, targets=["trip_duration"]):
    clms_x = [clm for clm in data.columns if clm not in targets]
    return data[clms_x], data[targets] if targets[0] in data.columns else None

def normalize(x, mn, mx):
    x = (x - mn) / ((mx - mn) + 1e-9)
    return x

def calc_normalize(x):
    mn, mx = np.min(x, axis=0), np.max(x, axis=0)
    return mn, mx

def calc_error(pred, y_true, model_params, name="val"):
    r2_val = r2_score(y_true, pred)
    mse_val = mean_squared_error(y_true, pred)
    model_params[f"r2_{name}"] = r2_val
    model_params[f"mse_{name}"] = mse_val
    return model_params