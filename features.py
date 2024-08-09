import pandas as pd
from constants import HOOP_LOCATIONS, MIDPOINT
from math import sqrt



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
    
    

"""
All functions below this comment must take at least the feature DF as a param
    and return a single column
"""

def get_distance_from_hoop(feature_df, locations):
    shot_locations = locations.loc[locations['annotation_code'] == 's']
    shot_locations['court_x_norm'] = shot_locations['court_x'].apply(normalize_to_hoop)
    shot_locations['court_y_norm'] = shot_locations['court_y']
    shot_locations['shot_distance'] = shot_locations.apply(dist_from_hoop, axis=1)
    shot_locations = shot_locations[['play_iid', 'shot_distance']]
    feature_df = feature_df.merge(right=shot_locations, how='left', left_on=['play_iid'], right_on=['play_iid'])
    return feature_df
    
    
    