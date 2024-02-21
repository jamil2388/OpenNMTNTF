'''
This file handles the tasks related to preprocessing data from opentf.
It will take the teamsvecs file and preprocess it to produce skills and member files
as source and target files for translation. We aim to translate from the list of
'Skills' to the list of 'Experts' in a team
'''

import pickle

# preprocess the teamsvecs file to produce desired translation files
def preprocess(filepath):
    teamsvecs = read_file(filepath)




def read_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            file = pickle.read(f)
        return file
    except FileNotFoundError:
        print(f'{filepath} not found')