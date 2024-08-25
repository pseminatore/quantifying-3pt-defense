from features import create_feature_df
from model import get_predictions, compare_features
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.options.mode.chained_assignment = None


def generate_submissions(model_file):
    feature_df, _ = create_feature_df(train=False)
    pred_df, score = get_predictions(model_file=model_file, rebuild_model=False, test_df=feature_df, feature_cols=['shot_distance'])


def compare_dist_columns():
    test_cols = [f'defenders_within_{n}_distance' for n in range(1, 15)]
    feature_df, test_df = create_feature_df(use_cache=True)
    best_feature, best_score = compare_features(feature_df, test_cols=test_cols, test_df=test_df)
    

def test_model():
    feature_df, test_df = create_feature_df(use_cache=True)
    pred_df, score = get_predictions(rebuild_model=True, feature_df=feature_df, feature_cols=['shot_distance', 'dist_from_nearest_defender', 'defenders_within_7_distance'], test_df=test_df, get_score=True)
    

def run():
    compare_dist_columns()
    

if __name__ == '__main__':
    run()