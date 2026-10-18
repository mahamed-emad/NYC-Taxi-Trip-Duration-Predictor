from src.model import *
from src.clean_data import *
from src.model_config import *
from src.preprocessing import *

def train(model_params, param_grid):
    x, y = split_data(remove_outliers(read("train"), *model_params["outliers"]))

    pip_lin = pipeline(model_params)
    grd_srch = grid_search(pip_lin, param_grid)
    print("prepared data")
    
    grd_srch.fit(x, log(y))
    model_params["alpha"] = grd_srch.best_params_["model__alpha"]
    print("trained model")
    
    return save_model(model_params, grd_srch.best_estimator_)