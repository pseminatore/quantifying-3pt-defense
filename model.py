import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix, accuracy_score
from os.path import exists
from constants import MODEL_OUTPUT_LOCATION, MODEL_SCORES_LOCATION
from time import time
import json
from sklearn.model_selection import GridSearchCV
from graphics import get_feature_importance
import numpy as np
from random import randint

def tune_hyperparams(feature_df, feature_cols, target_col):
    X = feature_df[feature_cols]
    y = feature_df[[target_col]]
    estimator = xgb.XGBClassifier(objective=custom_weighted_log_loss, booster='gbtree', eval_metric=roc_auc_score)
    parameters = {
        'max_depth': range (3, 10, 1),
        'n_estimators': range(50, 80, 10),
        'learning_rate': [0.01, 0.1, 0.001],
        'subsample': [0.7, 0.9, 1],
        'colsample_bytree': [0.6, 0.8, 1],
        'base_score': [0.35, 0.5]
    }
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        scoring = 'accuracy',
        n_jobs = 10,
        cv = 4,
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
    

def get_predictions(model_file='', rebuild_model=False, tune_model=False, feature_df=None, feature_cols=None, test_df=None, prod=False, get_score=False, save_model=True, show_feature_importance=True, model_comments=None):
    model, model_file = get_model(model_file, rebuild_model, feature_df, feature_cols, prod, save_model=save_model, tune_model=tune_model)
    pred_df = predict(model, test_df, feature_cols)
    score = None
    if get_score:
        score = score_predictions(pred_df, model, model_file, show_output=True, feature_cols=feature_cols, show_feature_importance=show_feature_importance, model_comments=model_comments)
    return pred_df, score


def score_predictions(pred_df, model, model_file, show_output=False, feature_cols=[], save_model=True, show_feature_importance=False, model_comments=None, show_confusion_matrix=True):
    score = roc_auc_score(pred_df['is_made'], pred_df['xMake'])
    loss_score = log_loss(pred_df['is_made'], pred_df['xMake'])
    cols = feature_cols + ['is_made', 'xMake']
    if show_output:
        print(f'ROC-AUC Score: {score:.4f}')
        print(f'Loss Score: {loss_score:.4f}')
    y_pred = pred_df.apply(lambda row: 1 if row['xMake'] > 0.46 else 0, axis=1)
    if show_confusion_matrix:
        plot_confusion_matrix(cm=confusion_matrix(pred_df['is_made'], y_pred), target_names=['Miss', 'Make'], normalize=False)
        
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
        xg_class = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', colsample_bynode=1, min_child_weight=1
                                    ,scale_pos_weight=1.71, **best_params)
    
    xg_class.fit(X_train,y_train, eval_metric=pct_correct_ranks, eval_set=[(X_train, y_train)])
    
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


def softmax(x):
    '''Softmax function with x as input vector.'''
    e = np.exp(x)
    return e / np.sum(e)

def pct_correct_ranks(y_pred: np.ndarray, y_true: xgb.DMatrix):
    corrects = 0
    y_vals = y_true.get_label()
    makes = np.nonzero(y_vals)[0]
    misses = np.where(y_vals == 0)[0]
    makes_range = len(makes)-1
    misses_range = len(misses)-1
    for idx, val in enumerate(y_pred):
        if y_vals[idx] == 1:
            comp_idx = misses[randint(0, misses_range)]
            comp_val = y_pred[comp_idx]
            if comp_val < val:
                corrects +=1
        else:
            comp_idx = makes[randint(0, makes_range)]
            comp_val = y_pred[comp_idx]
            if comp_val > val:
                corrects +=1
    result = corrects / len(y_pred) 
    return 'pctCorrectRanks', result

def custom_weighted_log_loss(y_true: np.ndarray, y_pred: np.ndarray):
    pred_prob = 1.0/ (1.0 + np.exp(-y_pred))
    alpha = 1
    grad = (pred_prob - y_true) * (y_true * alpha + (1-y_true))
    hess = pred_prob * (1 - pred_prob) * (y_true * alpha + (1 - y_true))
    return grad, hess
    

def rank_loss(y_true: np.ndarray, y_pred: np.ndarray):
    corrects = np.array([])
    preds = 1.0/ (1.0 + np.exp(-y_pred))
    y_vals = y_true
    makes = np.nonzero(y_vals)[0]
    misses = np.where(y_vals == 0)[0]
    makes_range = len(makes)-1
    misses_range = len(misses)-1
    for idx, val in enumerate(preds):
        if y_vals[idx] == 1:
            comp_idx = misses[randint(0, misses_range)]
            comp_val = preds[comp_idx]
            corrects = np.append(corrects, 0-val if val > comp_val else -1 + val)
        else:
            comp_idx = makes[randint(0, makes_range)]
            comp_val = preds[comp_idx]
            corrects = np.append(corrects, val if val < comp_val else 1 - val)
    grad = corrects * (y_true + (1 - y_true))
    hess = corrects * (1 - corrects) * (y_true + (1 - y_true))
    return grad, hess

def manual_rank(y_true: np.ndarray, y_pred: np.ndarray):
    corrects = np.array([])
    y_pred = softmax(y_pred)
    y_vals = y_true
    makes = np.nonzero(y_vals)[0]
    misses = np.where(y_vals == 0)[0]
    makes_range = len(makes)-1
    misses_range = len(misses)-1
    for idx, val in enumerate(y_pred):
        if y_vals[idx] == 1:
            comp_idx = misses[randint(0, misses_range)]
            comp_val = y_pred[comp_idx]
            corrects = np.append(corrects, 1 if comp_val > val else 0)
        else:
            comp_idx = makes[randint(0, makes_range)]
            comp_val = y_pred[comp_idx]
            corrects = np.append(corrects, 1 if comp_val < val else 0)
    grad = np.where(corrects < 1, -corrects / len(corrects), 0)      
    hess = np.repeat(1e-6, len(corrects))
    return grad, hess
    

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
    
    
def build_output_file(filename='submission.csv', pred_df=None):
    pred_df.rename(columns={'play_iid': 'ID', 'xMake': 'Target'}, inplace=True)
    pred_df = pred_df[['ID', 'Target']]
    pred_df.to_csv(filename, index=False)
    return