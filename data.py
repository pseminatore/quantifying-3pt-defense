import pandas as pd


def read_locations(train=True):
    pass 

def read_pbp(train=True):
    pass

def read_tracking(train=True):
    pass


def read_train_data(loc=True, pbp=True, tracking=True):
    """
    Read in training data
    
    Returns
    -----------
    dict : keys 'loc', 'pbp', 'tracking'
        each corresponding to DF or None with data file 
    
    """
    loc_data = read_locations() if loc else None
    pbp_data = read_pbp() if pbp else None
    tracking_data = read_tracking() if tracking else None
    
    ret_obj = {'loc': loc_data, 'pbp': pbp_data, 'tracking': tracking_data}
    
    
    return ret_obj