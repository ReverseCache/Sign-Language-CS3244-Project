import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("frame_data.csv")
filter_df = pd.read_csv("filter_list.csv")

df['Start'] = df['Start'].astype('int') - 1
df['End'] = df['End'].astype('int') - 1

labels = filter_df['Row Labels']

for label in labels:
    print("Extracting label: " + label)
    scene_list = []
    sessions_df = df.loc[df['Main'] == label]
    for row in sessions_df.itertuples():
        ind, main, session, scene, start, end = row
        scene_data = np.load(f'{session}/scene{scene}.npy')
        scene_list.append(scene_data)
    
    print("Saving scenes for label " + label)
    pickle.dump(scene_list, open(f'{label}_scenes.pkl', "wb"))