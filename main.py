from time import time
from src.train import *
from src.predict import *

model_params = {
    "models": "Ridge",
    "penalty": "l2",
    "eta0": 0.001,
    "fit_intercept": True,
    "max_iter": int(1e5),
    "alpha": 1,
    "tol": 5e-4,
    "batch_size": 32,
    "early_stopping": True,
    "validation_fraction": 0.05,
    "outliers": [0, 350000],
    "degree_poly": 2,
    "techniques": [[0.1], "grid-size=1e4"],
    "random_state": 42
}

start = time()
model_name = "model30"
model_name = train(model_params, param_grid={"model__alpha": model_params["techniques"][0]})

pred = test(model_name, read("train"), "train")
save(pred, "eda/pred", change=True)

test(model_name, read("val"), "val")
test(model_name, read("test"), "test")

test(model_name, read("train_kaggle"), "train_kaggle")
pred = test(model_name, read("test_kaggle"), "pred_kaggle")
save(pred, "pred_kaggle")

print(f"TIME-TAKEN: {(time() - start) / 60:.1f} mins.")