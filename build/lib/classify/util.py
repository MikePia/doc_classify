import os
import pandas as pd
import numpy as np
import pickle
from joblib import dump, load
import xgboost as xgb
from catboost import CatBoostClassifier

modelsdir = os.getenv("MODELS_DIR")
assert modelsdir is not None, "MODELS_DIR is not set"

def load_df_from_pickle(path):
    df = pd.read_pickle(path)
    return df


def load_np_array_from_pickle(path):
    np_array = np.load(path)
    return np_array


def load_vectorizer(path):
    vectorizer = pickle.load(open(path, "rb"))
    return vectorizer


models = {
    "forest": modelsdir + "/model1",
    "hgb": modelsdir + "/model2",
    "catboost": modelsdir + "/model3",
    "xgboost": modelsdir + "/model4",
}


def load_model(type):
    """
    Load the model from disk
    type = "forest" or "hgb" or "catboost" or "xgboost"
    """
    path = models[type]

    if type == "forest" or type == "hgb":
        model = load(path)
    elif type == "catboost":
        model = CatBoostClassifier()
        model.load_model(path)
    elif type == "xgboost":
        model = xgb.Booster()
    return model


def store_model(model, type):
    path = models[type]
    if type == "forest" or type == "hgb":
        dump(model, path)
    if type == "catboost":
        model.save_model(path)
    elif type == "xgboost":
        model.save_model(path)
