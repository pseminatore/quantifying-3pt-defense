from features import create_feature_df
from model import get_predictions, compare_features, build_output_file
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.options.mode.chained_assignment = None


def generate_submissions(model_file):
    feature_df, _ = create_feature_df(train=False, use_cache=False, cache_features=False)
    pred_df, score = get_predictions(model_file=model_file, rebuild_model=False, test_df=feature_df, feature_cols=['shot_distance', 'obfuscation_score', 'distance_traveled', 'shot_angle', 'time_within_3_feet'])
    build_output_file(pred_df=pred_df)


def compare_dist_columns():
    test_cols = [f'defenders_within_{n}_distance' for n in range(1, 15)]
    feature_df, test_df = create_feature_df(use_cache=True)
    best_feature, best_score = compare_features(feature_df, test_cols=test_cols, test_df=test_df)
    

def test_model():
    feature_df, test_df = create_feature_df(use_cache=True, cache_features=True)
    pred_df, score = get_predictions(rebuild_model=True, feature_df=feature_df, feature_cols=['shot_distance', 'obfuscation_score', 'distance_traveled', 'shot_angle', 'time_within_3_feet'], test_df=test_df, get_score=True, show_feature_importance=False, tune_model=True)
    

def run():
    generate_submissions('model/1727651525_0.txt')
    

if __name__ == '__main__':
    run()