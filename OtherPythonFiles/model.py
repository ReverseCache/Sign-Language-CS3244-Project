import pandas as pd
import pickle
object = pd.read_pickle(
    r'G:\code\Y2S1\CS3244\Data\col\collated\same_scenes.pkl')
print(len(object[1]))

'''
use this if you dont have pandas:

infile = open(r'G:\code\Y2S1\CS3244\Data\col\collated\afraid_scenes.pkl', 'rb')
array_values = pickle.load(infile)
# nesting as follow: Scene,Frame,DataPoints
infile.close()
print(len(array_values))
'''
#window
    # Frames
        # Frame
