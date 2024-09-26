from constants import HOOP_LOCATIONS, MIDPOINT, FEATURE_CACHE_LOCATION
from math import sqrt, atan2
from sklearn.model_selection import train_test_split
from data import read_data
import pandas as pd
from os.path import exists
from graphics import plot_obfuscation_scores
import numpy as np



def create_feature_df(train=True, test_size=0.2, cache_features=False, use_cache=False):
    if use_cache and exists(FEATURE_CACHE_LOCATION):
        feature_df = pd.read_csv(FEATURE_CACHE_LOCATION)
    else:
        data = read_data(tracking=False, train=train)
        feature_df = get_play_metadata(data['pbp'])
        feature_df = get_distance_from_hoop(feature_df, data['loc'])
        #feature_df = get_defenders_distance(feature_df, data['loc'])
        feature_df = get_obfuscation_score(feature_df, data['loc'])
        feature_df = get_distance_traveled(feature_df, data['tracking'])
    ########################################################################
    # Uncached Feature functions need to be added between these comments. 
    # Move function into the `else` block above once it has been cached.
    # Start new features
    ########################################################################
    #data = read_data(train=train)
    
    
    ########################################################################
    # End new features
    ########################################################################
    if cache_features:
            feature_df.to_csv(FEATURE_CACHE_LOCATION, index=False)
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
    return sqrt(((row['court_x'] - HOOP_LOCATIONS[side]['x']) ** 2) + ((row['court_y'] - HOOP_LOCATIONS[side]['y']) ** 2))
    


def dist_between_locations(origin, point):
    if origin.empty or point.empty:
        return None
    return sqrt(((origin['court_x'] - point['court_x']) ** 2) + ((origin['court_y'] - point['court_y']) ** 2))

    
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
    shot_locations['court_x'] = shot_locations['court_x'].apply(normalize_to_hoop)
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
        for dist in range(1, 15):
            feature_df.loc[feature_df['play_iid'] == play_iid, f'defenders_within_{dist}_distance'] = len(defender_locations.loc[defender_locations['dist_from_defender'] <= dist])
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

def get_distance_traveled(feature_df, tracking):
    tracking_df = tracking.copy()
    # Isolate shooter frames only
    tracking_df = tracking_df.loc[tracking_df['type'] == 'shooter']
    
    # Lag by a frame to get next frame on each row
    tracking_df['x_last'] = tracking_df['x'].shift(1)
    tracking_df['y_last'] = tracking_df['y'].shift(1)
    
    # Get euclidean distance between points
    tracking_df['delta'] = np.linalg.norm(tracking_df[['x', 'y']].values - tracking_df[['x_last', 'y_last']].values, axis=1)
    
    # Only want magnitude
    tracking_df['distance_traveled'] = tracking_df['delta'].abs()
    
    # Reduce to only necessary columns
    tracking_df = tracking_df[['play_iid', 'distance_traveled']]
    
    # Sum across frames on play to get total distance covered
    play_df = tracking_df.groupby(by='play_iid').sum()
    play_df.reset_index(inplace=True)
    
    # Join to feature DF and return
    feature_df = feature_df.merge(right=play_df, how='left', left_on=['play_iid'], right_on=['play_iid'])
    
    return feature_df
    
    
def get_obfuscation_score(feature_df, locations, dist_padding=3):
    locations_df = locations.copy()
    # locations_df = get_court_locations_df(24, 0)
    # locations_df = locations_df.loc[locations_df['play_iid'] == '19783001319551-6']
    
    # Only get necessary columns to keep pivoting straightforward
    locations_df = locations_df[['play_iid', 'annotation_code', 'court_x', 'court_y']]
    
    # Shooter and defenders only
    locations_df = locations_df.loc[locations_df['annotation_code'].isin(['s', 'd1', 'd2', 'd3', 'd4', 'd5'])]
    
    # Standardize to single hoop
    locations_df['court_x'] = locations_df['court_x'].apply(normalize_to_hoop)
    
    # Get everyones distance from hoop
    locations_df['dist'] = locations_df.apply(dist_from_hoop, axis=1)
    
    # Set origin to hoop location
    locations_df['court_x'] = locations_df['court_x'] - HOOP_LOCATIONS['L']['x']
    locations_df['court_y'] = locations_df['court_y'] - HOOP_LOCATIONS['L']['y']
    
    # Get everyones angle from hoop (in deg)
    locations_df['theta'] = (np.arctan2(locations_df['court_x'], locations_df['court_y']) * 180 / np.pi) - 90
    
    # Slice out shooter vs defender vars and smash them together
    #   Each row has shooter values and a single defenders values
    shooter_df = locations_df.loc[locations_df['annotation_code'] == 's']
    defenders_df = locations_df.loc[locations_df['annotation_code'] != 's']
    full_df = shooter_df.merge(defenders_df, on='play_iid', suffixes=('_s', '_d'))
    
    # Compute theta_3 as the difference between the two angles
    full_df['theta_3'] = full_df['theta_s'] - full_df['theta_d']
    
    # Take the cosine of theta_3 * defender distance from hoop to get the defenders
    #   distance along the axis of shooter to hoop (distance relative to direction of shot)
    full_df['def_dist_along_shot_axis'] = np.cos(np.deg2rad(full_df['theta_3'])) * full_df['dist_d']
    
    # Set the floor to 0 to avoid negative values
    full_df['def_dist_along_shot_axis'] = np.where(full_df['def_dist_along_shot_axis'] < 0, 0, full_df['def_dist_along_shot_axis'])

    
    # Handle case where defender is behind shooter relative to shot axis
    full_df['def_dist_along_shot_axis'] = np.where(full_df['def_dist_along_shot_axis'] > full_df['dist_s'], 0, full_df['def_dist_along_shot_axis'])
    
    # Apply distance padding
    #   *Distance padding accounts for defenders arms relative to eyeline of shooter to hoop
    ## TODO -- only apply distance padding when defender is in front of shooter
    full_df['adj_dist_along_shot_axis'] = full_df['def_dist_along_shot_axis'] + dist_padding
    
    # Get ratio of adj dist along axis to full shot
    full_df['def_dist_ratio'] = full_df['adj_dist_along_shot_axis'] / full_df['dist_s']
    
    # Square the ratio to scale
    full_df['sq_def_dist_ratio'] = full_df['def_dist_ratio'] ** 2
    
    # Get the defender offset distance
    full_df['def_offset_dist'] = np.sin(np.deg2rad(full_df['theta_3'])) * full_df['dist_d']
    
    # Get the remainder of defender distance to shooter
    full_df['def_dist_res'] = full_df['dist_s'] - full_df['def_dist_along_shot_axis']
    
    # Get the defender angle offset from shooter to hoop
    full_df['def_offset_angle'] = np.arctan(full_df['def_offset_dist']/full_df['def_dist_res'])  * 180 / np.pi
    
    # Calculate final obfuscation score
    full_df['obfuscation_score'] = (full_df['sq_def_dist_ratio'] * (1.2 * np.cos(np.deg2rad(full_df['def_offset_angle']))))# * 180 / np.pi
    
    # See what happens
    plot_obfuscation_scores(full_df)
    
    # Reduce to only necessary columns
    full_df = full_df[['play_iid', 'obfuscation_score']]
    
    # Sum across all defenders on each play
    play_df = full_df.groupby(by='play_iid').sum()
    play_df.reset_index(inplace=True)
    
    # Join to feature DF and return
    feature_df = feature_df.merge(right=play_df, how='left', left_on=['play_iid'], right_on=['play_iid'])
    return feature_df

    
    
    
    
    
    

    
    
    
    
def get_court_locations_df(shooter_x, shooter_y):
    xs = []
    ys = []
    play_iids = []
    annotation_codes = []
    for y in range(0, 51):
        for x in range(0, 91):
            xs.append(x)
            ys.append(y)
            play_iids.append(1)
            annotation_codes.append('d1')
    xs.append(shooter_x)
    ys.append(shooter_y)
    play_iids.append(1)
    annotation_codes.append('s')
    df = pd.DataFrame({'play_iid': play_iids, 
                       'annotation_code': annotation_codes,
                       'court_x': xs,
                       'court_y': ys,
                       })
    return df
    
        
    
    
    
def test_df():
    df = pd.DataFrame({'play_iid': [1, 1, 2, 2, 3, 3], 
                       'annotation_code': ['s', 'd', 's', 'd', 's', 'd'],
                       'court_x': [0, 1, 30, 29, 14, 15],
                       'court_y': [0, 1, 30, 29, 14, 15],
                       })
    return df
    
    