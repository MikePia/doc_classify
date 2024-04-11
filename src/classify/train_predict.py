from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
import xgboost as xgb

from classify.util import load_df_from_pickle, load_np_array_from_pickle, store_model


def split_data(features, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_model_random_forest(X_train, y_train, X_test, y_test):
    # Initialize the RandomForest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    print("Evaluating random forest model...")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return rf_classifier


def train_model_HistGradientBoosting(X_train, y_train, X_test, y_test):
    """
    Trains a HistGradientBoostingClassifier model and evaluates it.

    :param X_train: Training feature matrix
    :param y_train: Training target array
    :param X_test: Test feature matrix
    :param y_test: Test target array
    """

    # Initialize the HistGradientBoostingClassifier
    hgb_classifier = HistGradientBoostingClassifier(random_state=42)

    # Train the model
    hgb_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = hgb_classifier.predict(X_test)

    # Evaluate the model
    print("Evaluating HistGradientBoostingClassifier model...")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return hgb_classifier


def train_catboost(X_train, y_train, X_test, y_test):
    # Initialize the CatBoost Classifier
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        verbose=200,  # It prints the training log every 200 iterations
        random_state=42,
        eval_metric="Accuracy",  # You can change this to other metrics relevant to your task
    )

    # Train the model
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Evaluating CatBoostClassifier model...")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return model


def train_xgboost(X_train, y_train, X_test, y_test):
    # Adjust labels if they start from 1 instead of 0
    y_train = y_train - 1
    y_test = y_test - 1

    # Convert the datasets to DMatrix, which is a high-performance XGBoost data structure
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set up the parameters for XGBoost
    params = {
        "max_depth": 6,
        "eta": 0.3,
        "objective": "multi:softmax",  # Use softmax for multi-class classification
        "num_class": 3,  # Specify the number of unique classes
        "eval_metric": "mlogloss",  # Multiclass logloss for evaluation
    }
    num_rounds = 100

    # Train the model
    eval_set = [(dtrain, "train"), (dtest, "test")]
    bst = xgb.train(
        params, dtrain, num_rounds, evals=eval_set, early_stopping_rounds=10
    )

    # Make predictions
    y_pred = bst.predict(dtest)

    # Evaluate the model
    print("Evaluating XGBoost model...")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return bst


if __name__ == "__main__":
    # %% [markdown]
    # ## Implement Model Training

    # %%
    dff_pickle_path = "/dave/data/df_features.pkl"
    features_path = "/dave/data/features_array.pkl.npy"
    features = load_np_array_from_pickle(features_path)
    df = load_df_from_pickle(dff_pickle_path)

    X_train, X_test, y_train, y_test = split_data(features, df["presentation"])
    model1 = train_model_random_forest(X_train, y_train, X_test, y_test)
    model2 = train_model_HistGradientBoosting(X_train, y_train, X_test, y_test)
    model3 = train_catboost(X_train, y_train, X_test, y_test)
    model4 = train_xgboost(X_train, y_train, X_test, y_test)

    # %%
    models = [
        (model1, "/dave/data/model1", "forest"),
        (model2, "/dave/data/model2", "hgb"),
        (model3, "/dave/data/model3", "catboost"),
        (model4, "/dave/data/model4", "xgboost"),
    ]
    for model, path, type in models:
        store_model(model, type)

    # # Process and predict a single document
    # start = time.time()
    # single_prediction_model2 = model2.predict(doc_feature_array)
    # print("HistGradientBoosting Prediction:", single_prediction_model2)
    # print("Time taken:", time.time() - start)

    # start = time.time()
    # single_prediction_model3 = model3.predict(doc_feature_array)
    # print("CatBoost Prediction:", single_prediction_model3)
    # print("Time taken:", time.time() - start)

    # # %%
    # start = time.time()
    # dtest = xgb.DMatrix(doc_feature_array)

    # single_prediction_model4 = model4.predict(dtest)
    # print("XGBoost Prediction:", single_prediction_model4)

    # print("Time taken:", time.time() - start)
