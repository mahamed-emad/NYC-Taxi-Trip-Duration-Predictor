import numpy as np
import pandas as pd
from .functions import *
from .clean_data import *
from .model_config import *
from .preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer

def pipeline(model_params):
    categorical = [0, 1, 2]
    numerical = [3, 4, 5, 6, 7]
    preprocess = ColumnTransformer([
        ("catg", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", MinMaxScaler(), numerical)
    ])
    _pipeline = Pipeline(steps=[
        ("cln_clms", FunctionTransformer(clean_clms)),
        ("to_np", FunctionTransformer(to_np)),
        ("add_clms", FunctionTransformer(add_clms)),
        ("delete", FunctionTransformer(delete)),
        ("clms_trans", preprocess),
        ("poly", PolynomialFeatures(degree=model_params["degree_poly"], include_bias=False)),
        ("model", Ridge(**filter_params(Ridge, model_params)))
    ])
    return _pipeline

def grid_search(pipeline, param_grid):
    _grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=4,
        scoring="neg_root_mean_squared_error",
        n_jobs=3
    )
    return _grid_search