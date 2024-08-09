import pandas as pd
from data import read_data
from features import *
from model import get_model, predict


def get_predictions(model_file, rebuild_model=False, tune_model=False, feature_df=None, feature_cols=None, prod=False):
    model = get_model(model_file, rebuild_model, feature_df, feature_cols, prod)
    test_df = create_feature_df(train=False)
    pred_df = predict(model, test_df, feature_cols)
    return pred_df

def create_feature_df(train=True):
    data = read_data(tracking=False, train=train)
    feature_df = get_play_metadata(data['pbp'])
    feature_df = get_distance_from_hoop(feature_df, data['loc'])
    return feature_df

def generate_submissions():
    pass


def run():
    feature_df = create_feature_df()
    pred_df = get_predictions(rebuild_model=True, feature_df=feature_df, feature_cols=['shot_distance'])
    

if __name__ == '__main__':
    run()