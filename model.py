import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from os.path import exists
from constants import MODEL_OUTPUT_LOCATION, MODEL_SCORES_LOCATION
from time import time
import json

def tune_hyperparams():
    pass


def get_predictions(model_file='', rebuild_model=False, tune_model=False, feature_df=None, feature_cols=None, test_df=None, prod=False, get_score=False):
    model, model_file = get_model(model_file, rebuild_model, feature_df, feature_cols, prod)
    pred_df = predict(model, test_df, feature_cols)
    score = None
    if get_score:
        score = score_predictions(pred_df, model_file, show_output=True)
    return pred_df, score


def score_predictions(pred_df, model_file, show_output=False):
    score = roc_auc_score(pred_df['is_made'], pred_df['xMake'])
    if show_output:
        print(f'ROC-AUC Score: {score:.2f}')
        
    with open(MODEL_SCORES_LOCATION, "r+") as scores_file:
        scores_obj = json.load(scores_file)
    scores_obj[model_file] = score
    with open(MODEL_SCORES_LOCATION, "w+") as scores_file:
        json.dump(scores_obj, scores_file, indent=4)
    return score

def predict(model, test_df, feature_cols):
    X_test = test_df[feature_cols]
    pred_prob = model.predict_proba(X_test)
    preds = [pred[1] for pred in pred_prob]
    test_df['xMake'] = preds
    return test_df

def get_model(model_file, rebuild_model, df=None, feature_cols=None, prod=False):
    if not exists(model_file) or rebuild_model:
        if df is None:
            raise Exception('Missing Required params')
        model, model_file = build_model(df, feature_cols, prod)
    else:
        model = xgb.XGBRegressor()
        model.load_model(model_file)
    return model, model_file

def build_model(df, feature_cols=[], prod=False):
    X_train = df[feature_cols]
    y_train = df[['is_made']]
    xg_class = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', eval_metric=roc_auc_score, learning_rate=0.2, gamma=0
                                , subsample=0.8, colsample_bytree=0.8, colsample_bynode=1, max_depth=4, min_child_weight=1)

    xg_class.fit(X_train,y_train)
    
    model_file = save_model(xg_class, prod)
    
    return xg_class, model_file

def save_model(model, prod):
    iteration = f'{time():.0f}'
    is_prod = 1 if prod else 0
    model_filename = MODEL_OUTPUT_LOCATION.format(iteration=iteration, prod=is_prod)
    model.save_model(model_filename)
    return model_filename
