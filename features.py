from constants import HOOP_LOCATIONS, MIDPOINT
from math import sqrt
from sklearn.model_selection import train_test_split
from data import read_data
import numpy as np


def create_feature_df(train=True, test_size=0.2):
    data = read_data(tracking=False, train=train)
    feature_df = get_play_metadata(data['pbp'])
    feature_df = get_distance_from_hoop(feature_df, data['loc'])
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
    
    
def get_shooter_velocity(frame_num, df_tracking, frames_bef, frames_aft):
    basket_x, basket_y = 4, 25

    frames = range(frame_num - frames_bef, frame_num + frames_aft + 1)
    df_frames = df_tracking[df_tracking['frame'].isin(frames)]
    
    if df_frames.empty or len(df_frames) < 2:
        return np.nan, np.nan
    
      # Calculate distances between consecutive frames to compute speed
    df_frames['dx'] = df_frames['x_smooth'].diff()
    df_frames['dy'] = df_frames['y_smooth'].diff()
    
    df_frames['time_diff'] = df_frames['frame'].diff() / 30  # time difference in seconds
    df_frames['vx'] = df_frames['dx'] / df_frames['time_diff']
    df_frames['vy'] = df_frames['dy'] / df_frames['time_diff']
    
    avg_vx = df_frames['vx'].mean()
    avg_vy = df_frames['vy'].mean()

    
    # Calculate the angle of the velocity vector relative to the basket position
    direction_to_basket_x = basket_x - df_frames.iloc[-1]['x_smooth']
    direction_to_basket_y = basket_y - df_frames.iloc[-1]['y_smooth']
    
    # Angle between velocity vector and direction to basket
    dot_product = avg_vx * direction_to_basket_x + avg_vy * direction_to_basket_y
    magnitude_velocity = np.sqrt(avg_vx**2 + avg_vy**2)
    magnitude_basket_direction = np.sqrt(direction_to_basket_x**2 + direction_to_basket_y**2)
    
    # Compute the angle in radians and convert to degrees
    cos_theta = dot_product / (magnitude_velocity * magnitude_basket_direction)

    speed = np.sqrt(avg_vx**2 + avg_vy**2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    start_pos = df_frames['x_smooth'].iloc[0], df_frames['y_smooth'].iloc[0]
    end_pos = df_frames['x_smooth'].iloc[-1],df_frames['y_smooth'].iloc[-1]
    return speed, angle,start_pos,end_pos



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
    
    

    