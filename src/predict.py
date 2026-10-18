from src import *

def test(model_name, data, name=None):
    x, y = split_data(data)
    result = pd.DataFrame({
        "id": x["id"]
    })
    x = x.drop(columns=["dropoff_datetime"], errors="ignore")
    model = open_model(model_name)
    pred = model.predict(x)
    print("predicted")
    if y is not None:
        print(np.column_stack((pred, log(y))))
        model_params = get_hyperparameters(model_name)
        model_params = calc_error(pred, log(y), model_params, name)
        save_model(model_params, name=model_name)
    result["trip_duration"] = exp(pred)
    return result