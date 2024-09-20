import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from os.path import exists
from constants import MODEL_OUTPUT_LOCATION, MODEL_SCORES_LOCATION
from time import time
import json
from sklearn.model_selection import GridSearchCV
from graphics import get_feature_importance


def tune_hyperparams(feature_df, feature_cols, target_col):
    X = feature_df[feature_cols]
    y = feature_df[[target_col]]
    estimator = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', eval_metric=roc_auc_score)
    parameters = {
        'max_depth': range (3, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.025, 0.05, 0.1, 0.2],
        #'subsample': [0.7, 0.9, 1],
        #'colsample_bytree': [0.8, 1],
        'gamma': [0, 1, 2, 3],
        'base_score': [0.35, 0.5]
    }
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        scoring = 'accuracy',
        n_jobs = 10,
        cv = 2,
        verbose=True
    )
    grid_search.fit(X, y)
    print('best estimator: %s' % grid_search.best_params_)
    return grid_search.best_params_


def compare_features(static_df=None, test_cols=[], static_cols=['shot_distance', 'dist_from_nearest_defender'], test_df=None, print_results=True):
    tested_features = {}
    for col in test_cols:
        feature_cols = static_cols + [col] 
        feature_df = static_df[feature_cols + ['is_made']]
        _, score = get_predictions(rebuild_model=True, feature_df=feature_df, feature_cols=feature_cols, test_df=test_df, get_score=True, save_model=False)
        tested_features[col] = score
    best_feature = max(tested_features, key=tested_features.get)
    best_score = tested_features[best_feature]
    if print_results:
        print(tested_features)
        print(f'Best Feature: {best_feature} - {best_score:.4f}')
    return best_feature, best_score
    

def get_predictions(model_file='', rebuild_model=False, tune_model=False, feature_df=None, feature_cols=None, test_df=None, prod=False, get_score=False, save_model=True, show_feature_importance=False, model_comments=None):
    model, model_file = get_model(model_file, rebuild_model, feature_df, feature_cols, prod, save_model=save_model, tune_model=tune_model)
    pred_df = predict(model, test_df, feature_cols)
    score = None
    if get_score:
        score = score_predictions(pred_df, model, model_file, show_output=True, feature_cols=feature_cols, show_feature_importance=show_feature_importance, model_comments=model_comments)
    return pred_df, score


def score_predictions(pred_df, model, model_file, show_output=False, feature_cols=[], save_model=True, show_feature_importance=False, model_comments=None):
    score = roc_auc_score(pred_df['is_made'], pred_df['xMake'])
    loss_score = log_loss(pred_df['is_made'], pred_df['xMake'])
    cols = feature_cols + ['is_made', 'xMake']
    inspect_df = pred_df[cols]
    if show_output:
        print(f'ROC-AUC Score: {score:.4f}')
        print(f'Loss Score: {loss_score:.4f}')
        
    if show_feature_importance:
        get_feature_importance(model)
    if save_model:
        with open(MODEL_SCORES_LOCATION, "r+") as scores_file:
            scores_obj = json.load(scores_file)
        model_metadata = {}
        model_metadata['score'] = score
        model_metadata['loss'] = loss_score
        model_metadata['feature_cols'] = feature_cols
        if model_comments is not None:
            model_metadata['comments'] = model_comments
        scores_obj[model_file] = model_metadata
        with open(MODEL_SCORES_LOCATION, "w+") as scores_file:
            json.dump(scores_obj, scores_file, indent=4)
    return score


def predict_static(test_df, static_guess=0.35):
    test_df['xMake'] = static_guess
    return test_df


def predict(model, test_df, feature_cols, absolute=False, absolute_cutoff=0.35):
    X_test = test_df[feature_cols]
    if absolute:
        if absolute_cutoff is None:
            preds = model.predict(X_test)
        else:
            pred_prob = model.predict_proba(X_test) 
            preds = [1 if pred[1] > absolute_cutoff else 0 for pred in pred_prob]
    else:
        pred_prob = model.predict_proba(X_test) 
        preds = [pred[1] for pred in pred_prob]
    test_df['xMake'] = preds
    return test_df

def get_model(model_file, rebuild_model, df=None, feature_cols=None, prod=False, save_model=True, tune_model=False):
    if not exists(model_file) or rebuild_model:
        if df is None:
            raise Exception('Missing Required params')
        model, model_file = build_model(df, feature_cols, prod, save_model, tune_model)
    else:
        model = xgb.XGBClassifier()
        model.load_model(model_file)
    return model, model_file

def build_model(df, feature_cols=[], prod=False, save_model=True, tune_model=False):
    X_train = df[feature_cols]
    y_train = df[['is_made']]
    if not tune_model:
        xg_class = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', eval_metric=log_loss, learning_rate=0.2, gamma=0
                                    ,  colsample_bynode=1, max_depth=4, min_child_weight=1)
    else:
        best_params = tune_hyperparams(df, feature_cols, 'is_made')
        xg_class = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', eval_metric=log_loss, base_score=0.3
                                    , subsample=0.7, colsample_bytree=1.0, colsample_bynode=1, min_child_weight=1, **best_params)
    
    xg_class.fit(X_train,y_train)
    
    if save_model:
        model_file = write_model(xg_class, prod)
    else: 
        model_file = None
    
    return xg_class, model_file

def write_model(model, prod):
    iteration = f'{time():.0f}'
    is_prod = 1 if prod else 0
    model_filename = MODEL_OUTPUT_LOCATION.format(iteration=iteration, prod=is_prod)
    model.save_model(model_filename)
    return model_filename


def get_confusion_matrix():
    pass
