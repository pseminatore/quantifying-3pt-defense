import pandas as pd
from constants import FILE_LOCATIONS
import numpy as np


def read_locations(train=True):
    file_path = FILE_LOCATIONS['train']['locations'] if train else FILE_LOCATIONS['test']['locations']
    # Specify data types to speed up reading
    dtype = {
        'game_id': 'int64',
        'play_id': 'int32',
        'annotation_code': 'category',
        'court_x': 'float64',
        'court_y': 'float64'
    }
    
    # Usecols to only load necessary columns
    usecols = ['game_id', 'play_id', 'annotation_code', 'court_x', 'court_y']
    
    df = pd.read_csv(file_path, dtype=dtype, usecols=usecols)
    df['play_iid'] = df['game_id'].astype(str) + '-' + df['play_id'].astype(str)
    return df

def read_pbp(train=True):
    file_path = FILE_LOCATIONS['train']['pbp'] if train else FILE_LOCATIONS['test']['pbp']
    df = pd.read_csv(file_path)
    df['play_iid'] = df['game_id'].astype(str) + '-' + df['play_id'].astype(str)
    return df

def read_tracking(train=True):
    file_path = FILE_LOCATIONS['train']['tracking'] if train else FILE_LOCATIONS['test']['tracking']
    # Specify data types
    dtype = {
        'game_id': 'int64',
        'play_id': 'int32',
        'type': 'category',
        'x': 'float32',
        'y': 'float32',
        'frame': 'int32'
        # add other columns and their types as necessary
    }
    
    usecols = ['game_id', 'play_id', 'type', 'x', 'y', 'frame']  # add other columns as necessary
    
    df = pd.read_csv(file_path, dtype=dtype, usecols=usecols)
    df['play_iid'] = df['game_id'].astype(str) + '-' + df['play_id'].astype(str)
    return df

def read_data(locations=True, pbp=True, tracking=True, train=True):
    """
    Read in training data
    
    Returns
    -----------
    dict : keys 'loc', 'pbp', 'tracking'
        each corresponding to DF or None with data file 
    
    """
    loc_data = read_locations(train) if locations else None
    pbp_data = read_pbp(train) if pbp else None
    tracking_data = read_tracking(train) if tracking else None
    
    ret_obj = {'loc': loc_data, 'pbp': pbp_data, 'tracking': tracking_data}
    
    
    return ret_obj


def process_all_games_and_plays():
    results = []
    df_loc = read_locations()
    print("locations read in")
    df_tracking = read_tracking()
    print("tracking read in")

    for (game_id, play_id), df_play_loc in df_loc.groupby(['game_id', 'play_id']):
        print(game_id, play_id)
        # Load the tracking data for the current game and play
        df_play = df_tracking[(df_tracking['game_id'] == game_id) & (df_tracking['play_id'] == play_id)]
        df_offense = df_play[df_play['type'] == 'teammate'].copy()
        df_shooter = df_play[df_play['type'] == 'shooter'].copy()
        
        shooter = df_play_loc[df_play_loc['annotation_code'] == 's']
        teammates = df_play_loc[df_play_loc['annotation_code'].isin(['t1', 't2', 't3', 't4'])]

        # Find the best frame
        best_frame, min_distance = find_best_frame(df_offense, df_shooter, teammates, shooter)
        print(best_frame)

        # Store the result
        results.append({'game_id': game_id, 'play_id': play_id, 'frame': best_frame, 'min_distance': min_distance})

    # Convert the results to a DataFrame and save as a CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('best_frames.csv', index=False)


def get_play_df(game_id,play_id):
    df = read_tracking()
    df_play = df[(df['game_id'] == game_id) & (df['play_id'] == play_id)]
    df_offense = df_play[df_play['type'] == 'teammate']
    df_defense = df_play[df_play['type'] == 'defender']
    df_shooter = df_play[df_play['type'] == 'shooter']

    return df_offense, df_defense, df_shooter




def find_best_frame(df_offense, df_shooter, teammates, shooter):
    """
    Find the frame where all players are closest to their shooting positions.
    
    :param df_offense: DataFrame containing offensive player positions
    :param df_shooter: DataFrame containing shooter positions
    :param teammates: DataFrame containing teammate shot locations
    :return: The frame where the sum of distances for all players is minimized
    """

    if df_offense.empty or df_shooter.empty or shooter.empty:
        # If any relevant DataFrame is empty, return NaN for the best frame and minimum distance
        return np.nan, np.nan
    
    valid_frames = df_shooter.dropna(subset=['x', 'y'])['frame'].unique()
    
    if len(valid_frames) == 0:
        # If no valid frames are left after filtering, return NaN
        return np.nan, np.nan

    
    # Filter df_offense to only include valid frames
    df_offense = df_offense[df_offense['frame'].isin(valid_frames)].copy()
    df_shooter = df_shooter[df_shooter['frame'].isin(valid_frames)].copy()

    # Vectorized distance calculations for each teammate
    distances = []
    for _, teammate in teammates.iterrows():
        distances.append(np.sqrt((df_offense['x_smooth'] - teammate['court_x']) ** 2 + 
                                 (df_offense['y_smooth'] - teammate['court_y']) ** 2))

    if distances:
        df_offense.loc[:, 'min_distance'] = np.min(distances, axis=0)  # Use loc to avoid SettingWithCopyWarning
    else:
        df_offense.loc[:, 'min_distance'] = np.nan


    # Vectorized distance calculation for the shooter
    shooter_position = (shooter['court_x'].values[0], shooter['court_y'].values[0])
    df_shooter.loc[:, 'distance_to_shot'] = np.sqrt((df_shooter['x'] - shooter_position[0]) ** 2 + 
                                                    (df_shooter['y'] - shooter_position[1]) ** 2)

    # Sum distances for each frame
    if not df_offense.empty and not df_shooter.empty:
        total_distances = df_offense.groupby('frame')['min_distance'].sum() + \
                          df_shooter.groupby('frame')['distance_to_shot'].sum()

        # Find the frame with the minimum total distance
        best_frame = total_distances.idxmin()
        min_distance = total_distances.min()
    else:
        best_frame = np.nan
        min_distance = np.nan
    
    return best_frame, min_distance
    

if __name__ == '__main__':
    process_all_games_and_plays()
