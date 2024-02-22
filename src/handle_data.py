'''
This file handles the tasks related to preprocessing data from opentf.
It will take the teamsvecs file and preprocess it to produce skills and member files
as source and target files for translation. We aim to translate from the list of
'Skills' to the list of 'Experts' in a team
'''
import pickle
import os

# preprocess the teamsvecs file to produce desired translation files
# for example : one row of teamsvecs matrix - id : [5] skill : [0 0 1 0 1 0 0] member : [0 1 0 0 1]
# this represents the team 5 contains members 2 and 5 and skills 3 and 5
# now the desired output is
# src-train.txt : s3 s5
# tgt-tran.txt : m2 m5
def preprocess(filepath):
    teamsvecs = read_file(filepath)
    src = ["" for _ in range(teamsvecs['id'].shape[0])] # populate with skill corresponding to the team_id
    tgt = ["" for _ in range(teamsvecs['id'].shape[0])]

    # skill -> src
    # row = team_id (0 indexed)
    # col = skill_id (0 indexed)
    src = create_sentences('skill', 's', src, teamsvecs)
    tgt = create_sentences('member', 'm', tgt, teamsvecs)

    return src, tgt

# generate sentences based on the mode (skill or member)
def create_sentences(mode, chr, arr, teamsvecs):
    prev_row = 0
    tmp_arr = []
    for row, col in zip(teamsvecs[mode].nonzero()[0], teamsvecs[mode].nonzero()[1]):
        if(row != prev_row):
            if len(tmp_arr) > 0: arr[prev_row] = ' '.join(tmp_arr) # store the skills in the desired row
            prev_row = row
            tmp_arr = []
        tmp_arr.append(f"{chr}{col}")
    arr[prev_row] = ' '.join(tmp_arr)
    return arr

def read_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            file = pickle.load(f)
        if file is not None : print(f'{filepath} read successfully')
        return file
    except FileNotFoundError:
        print(f'{filepath} not found')


# overwrites existing file
def write_txt_file(data, filepath):
    dir = os.path.split(filepath)[0]
    if not os.path.exists(dir) : os.makedirs(dir)

    with open(filepath, 'w') as f:
        for line in data:
            f.write(line + '\n')
    print(f'{filepath} saved successfully')