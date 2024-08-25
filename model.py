import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from os.path import exists
from constants import MODEL_OUTPUT_LOCATION, MODEL_SCORES_LOCATION
from time import time
import json
from sklearn.model_selection import GridSearchCV


def tune_hyperparams(feature_df, feature_cols, target_col):
    X = feature_df[feature_cols]
    y = feature_df[[target_col]]
    estimator = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', eval_metric=log_loss)
    parameters = {
        'max_depth': range (2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05],
        #'eta': [0.0125, 0.025, 0.05],
        #'subsample': [0.7, 0.8, 0.9, 1],
        'colsample_bytree': [0.8, 1],
        #'gamma': [0, 1, 2, 3]
    }
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        scoring = 'r2',
        n_jobs = 10,
        cv = 10,
        verbose=True
    )
    grid_search.fit(X, y)
    print('best estimator: %s' % grid_search.best_estimator_)


def compare_features(static_df=None, test_cols=[], static_cols=['shot_distance', 'dist_from_nearest_defender'], test_df=None, print_results=True):
    tested_features = {}
    for col in test_cols:
        feature_cols = static_cols + [col]
        feature_df = static_df[feature_cols]
        _, score = get_predictions(rebuild_model=True, feature_df=feature_df, feature_cols=feature_cols, test_df=test_df, get_score=True)
        tested_features[col] = score
    best_feature = max(tested_features, key=tested_features.get)
    best_score = tested_features[best_feature]
    if print_results:
        print(tested_features)
    return best_feature, best_score
    

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


def predict_static(test_df, static_guess=0.35):
    test_df['xMake'] = static_guess
    return test_df


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
        model = xgb.XGBClassifier()
        model.load_model(model_file)
    return model, model_file

def build_model(df, feature_cols=[], prod=False):
    X_train = df[feature_cols]
    y_train = df[['is_made']]
    xg_class = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', eval_metric=log_loss, learning_rate=0.2, gamma=0
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
