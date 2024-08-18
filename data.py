import pandas as pd
from constants import FILE_LOCATIONS


def read_locations(train=True):
    file_path = FILE_LOCATIONS['train']['locations'] if train else FILE_LOCATIONS['test']['locations']
    df =  pd.read_csv(file_path)
    df['play_iid'] = df['game_id'].astype(str) + '-' + df['play_id'].astype(str)
    return df

def read_pbp(train=True):
    file_path = FILE_LOCATIONS['train']['pbp'] if train else FILE_LOCATIONS['test']['pbp']
    df = pd.read_csv(file_path)
    df['play_iid'] = df['game_id'].astype(str) + '-' + df['play_id'].astype(str)
    return df

def read_tracking(train=True):
    file_path = FILE_LOCATIONS['train']['tracking'] if train else FILE_LOCATIONS['test']['tracking']
    df = pd.read_csv(file_path)
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