HOOP_LOCATIONS = {
    'L': {
        'x': 4, 
        'y': 25
        }, 
    'R': {
        'x': 90, 
        'y': 25
        }
    }

MIDPOINT = {
    'x': 47,
    'y': 25
}

FILE_LOCATIONS = {
    'train': {
        'pbp': 'data/train_pbp.csv',
        'locations': 'data/train_locations.csv',
        'tracking': 'data/train_tracking.csv'
        },
    'test': {
        'pbp': 'data/test_pbp.csv',
        'locations': 'data/test_locations.csv',
        'tracking': 'data/test_tracking.csv'
    }
}

MODEL_OUTPUT_LOCATION = 'model/{iteration}_{prod}.txt'   

MODEL_SCORES_LOCATION = 'model/model_scores.json' 

FEATURE_CACHE_LOCATION = 'processed_features.csv'