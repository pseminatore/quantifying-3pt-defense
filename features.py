from constants import HOOP_LOCATIONS, MIDPOINT
from math import sqrt
from sklearn.model_selection import train_test_split
from data import read_data


def create_feature_df(train=True, test_size=0.2):
    data = read_data(tracking=False, train=train)
    feature_df = get_play_metadata(data['pbp'])
    feature_df = get_distance_from_hoop(feature_df, data['loc'])
    feature_df = get_defenders_distance(feature_df, data['loc'])
    #feature_df = get_distance_from_nearest_defender(feature_df, data['loc'])
    #feature_df = get_defenders_within_n_distance(feature_df, data['loc'])
    if train:
        feature_df, test_df = train_test_split(feature_df, test_size=test_size, random_state=42)
    else:
        test_df = None
    return feature_df, test_df


def get_play_metadata(pbp):
    df = pbp.copy()
    df.rename(columns={'Target': 'is_made'}, inplace=True)
    return df

def normalize_to_hoop(x, side='L'):
    """
    `x` should be x locations from `court_x`
    """
    if side == 'L':
        if x <= MIDPOINT['x']:
            return x
        else:
            return (HOOP_LOCATIONS['R']['x'] - x) + HOOP_LOCATIONS['L']['x']
    elif side == 'R':
        if x > MIDPOINT['x']:
            return x
        else:
            return HOOP_LOCATIONS['L']['x'] - (x - HOOP_LOCATIONS['R']['x'])
    else:
        return None
        

def dist_from_hoop(row, side='L'):
    return sqrt(((row['court_x_norm'] - HOOP_LOCATIONS[side]['x']) ** 2) + ((row['court_y_norm'] - HOOP_LOCATIONS[side]['y']) ** 2))
    

def dist_between_locations(origin, point):
    if origin.empty or point.empty:
        return None
    return sqrt(((origin['court_x'] - point['court_x']) ** 2) + ((origin['court_y'] - point['court_y']) ** 2))


"""
All functions below this comment must take at least the feature DF as a param
    and return the same feature DF with the added feature column.
    
The format of these feature columns is subject to change.
"""

def get_distance_from_hoop(feature_df, locations):
    shot_locations = locations.loc[locations['annotation_code'] == 's']
    shot_locations['court_x_norm'] = shot_locations['court_x'].apply(normalize_to_hoop)
    shot_locations['court_y_norm'] = shot_locations['court_y']
    shot_locations['shot_distance'] = shot_locations.apply(dist_from_hoop, axis=1)
    shot_locations = shot_locations[['play_iid', 'shot_distance']]
    feature_df = feature_df.merge(right=shot_locations, how='left', left_on=['play_iid'], right_on=['play_iid'])
    return feature_df



def get_defenders_distance(feature_df, locations, n=7):
    play_iids = feature_df['play_iid'].unique().tolist()
    for play_iid in play_iids:
        shooter_location = locations.loc[(locations['play_iid'] == play_iid) & (locations['annotation_code'] == 's')]
        defender_locations = locations.loc[(locations['play_iid'] == play_iid) & (locations['annotation_code'].isin(['d1', 'd2', 'd3', 'd4', 'd5']))]
        defender_locations.loc[:,'dist_from_defender'] = defender_locations.apply(dist_between_locations, axis=1, point=shooter_location)
        feature_df.loc[feature_df['play_iid'] == play_iid, 'dist_from_nearest_defender'] = defender_locations['dist_from_defender'].min()
        feature_df.loc[feature_df['play_iid'] == play_iid, 'defenders_within_12_distance'] = len(defender_locations.loc[defender_locations['dist_from_defender'] <= 12])
        feature_df.loc[feature_df['play_iid'] == play_iid, 'defenders_within_7_distance'] = len(defender_locations.loc[defender_locations['dist_from_defender'] <= n])
    return feature_df
    
def get_defenders_within_n_distance(feature_df, locations, n=7):
    play_iids = feature_df['play_iid'].unique().tolist()
    for play_iid in play_iids:
        shooter_location = locations.loc[(locations['play_iid'] == play_iid) & (locations['annotation_code'] == 's')]
        defender_locations = locations.loc[(locations['play_iid'] == play_iid) & (locations['annotation_code'].isin(['d1', 'd2', 'd3', 'd4', 'd5']))]
        defender_locations.loc[:, 'dist_from_defender'] = defender_locations.apply(dist_between_locations, axis=1, point=shooter_location)
        feature_df.loc[feature_df['play_iid'] == play_iid, 'defenders_within_n_distance'] = len(defender_locations.loc[defender_locations['dist_from_defender'] <= n])
    return feature_df
    
def get_distance_from_nearest_defender(feature_df, locations):
    play_iids = feature_df['play_iid'].unique().tolist()
    for play_iid in play_iids:
        shooter_location = locations.loc[(locations['play_iid'] == play_iid) & (locations['annotation_code'] == 's')]
        defender_locations = locations.loc[(locations['play_iid'] == play_iid) & (locations['annotation_code'].isin(['d1', 'd2', 'd3', 'd4', 'd5']))]
        defender_locations.loc[:,'dist_from_defender'] = defender_locations.apply(dist_between_locations, axis=1, point=shooter_location)
        feature_df.loc[feature_df['play_iid'] == play_iid, 'dist_from_nearest_defender'] = defender_locations['dist_from_defender'].min()
    return feature_df
    
    