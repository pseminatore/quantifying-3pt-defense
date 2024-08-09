import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from os.path import exists
from constants import MODEL_OUTPUT_LOCATION
from time import time

def tune_hyperparams():
    pass

def predict(model, test_df, feature_cols):
    X_test = test_df[feature_cols]
    # y_test = test_df[['is_made']]
    pred_prob = model.predict_proba(X_test)
    preds = [pred[1] for pred in pred_prob]
    test_df['xMake'] = preds
    return test_df

def get_model(model_file, rebuild_model, df=None, feature_cols=None, prod=False):
    if not exists(model_file) or rebuild_model:
        if not df:
            raise Exception('Missing Required params')
        model = build_model(df, feature_cols, prod)
    else:
        model = xgb.XGBRegressor()
        model.load_model(model_file)
    return model

def build_model(df, feature_cols=[], prod=False):
    X_train = df[feature_cols]
    y_train = df[['is_made']]
    xg_class = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', eval_metric=log_loss, learning_rate=0.2, gamma=0
                                , subsample=0.8, colsample_bytree=0.8, colsample_bynode=1, max_depth=4, min_child_weight=1)

    xg_class.fit(X_train,y_train)
    
    save_model(xg_class, prod)
    
    return xg_class

def save_model(model, prod):
    iteration = time()
    is_prod = 1 if prod else 0
    model.save_model(MODEL_OUTPUT_LOCATION.format(iteration=iteration, prod=is_prod))

def get_predictions(model, test_df):
    pass

def generate_submission():
    pass