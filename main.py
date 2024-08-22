from features import create_feature_df
from model import get_predictions
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.options.mode.chained_assignment = None


def generate_submissions(model_file):
    feature_df, _ = create_feature_df(train=False)
    pred_df, score = get_predictions(model_file=model_file, rebuild_model=False, test_df=feature_df, feature_cols=['shot_distance'])

def test_model():
    feature_df, test_df = create_feature_df()
    pred_df, score = get_predictions(rebuild_model=True, feature_df=feature_df, feature_cols=['shot_distance', 'dist_from_nearest_defender', 'defenders_within_12_distance'], test_df=test_df, get_score=True)
    

def run():
    test_model()
    

if __name__ == '__main__':
    run()