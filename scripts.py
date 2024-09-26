import pandas as pd




def run():
    features_df = pd.read_csv('processed_features.csv')
    binned_distance = pd.cut(features_df['shot_distance'],[24,26,28,30,90],labels = [1,2,3,4])
    #print(binned_distance)
    features_df['binned_distance'] = binned_distance
    distance_shooting_perc  = features_df[['is_made','binned_distance']].groupby(by='binned_distance').mean()
    #print(distance_shooting_perc)


    binned_obf= pd.cut(features_df['obfuscation_score'],[0,0.5,1,1.5,2,2.5,100],labels = [1,2,3,4,5,6])
    #print(binned_distance)
    features_df['binned_obf'] = binned_obf
    obf_shooting_perc  = features_df[['is_made','binned_obf']].groupby(by='binned_obf').mean()
    #print(obf_shooting_perc)


    binned_trav= pd.cut(features_df['distance_traveled'],[0,50,100,150,200,250,300,10000],labels = [1,2,3,4,5,6,7])
    #print(binned_distance)
    features_df['distance_traveled'] = binned_trav
    trav_shooting_perc  = features_df[['is_made','distance_traveled']].groupby(by='distance_traveled').mean()
    print(trav_shooting_perc)

    features_df.to_csv('processed_features.csv',index=False)

    


if __name__ == '__main__':
    run()