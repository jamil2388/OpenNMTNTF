'''
For handling the pipeline for entire translation framework
'''

import handle_data
from sklearn.model_selection import train_test_split

# global variables
random_state = 10

if __name__ == '__main__':

    domains = ['dblp/toy.dblp.v12.json']

    for domain in domains:

        # the paths here consider that the main.py is run from its native folder 'src'
        # so 'data' folder will be one up like '../data'
        # filepath : the path of the file
        # path : the path of the directory
        teamsvecs_filepath = f'../data/raw/{domain}/teamsvecs.pkl'
        preprocessed_path = f'../data/preprocessed/{domain}'
        split_path = f'../data/input/{domain}' # path to store the train-test-val splits

        preprocesed_source_filepath = f'{preprocessed_path}/src.txt' # these are the base files
        preprocesed_target_filepath = f'{preprocessed_path}/tgt.txt'

        # preprocess the data to build translation corpus in terms of skills and members
        # src and tgt are arrays of strings where each line is like "s0 s7 s10" etc.
        src, tgt = handle_data.preprocess(teamsvecs_filepath)

        # create splits
        src_train, src_test, tgt_train, tgt_test = train_test_split(src, tgt, test_size=0.2, random_state=random_state)
        src_train, src_val, tgt_train, tgt_val = train_test_split(src_train, tgt_train, test_size=0.2, random_state=random_state)

        ## write the arrays into files
        # base files first
        handle_data.write_txt_file(src, f"{preprocesed_source_filepath}")
        handle_data.write_txt_file(tgt, f"{preprocesed_target_filepath}")

        # splits
        handle_data.write_txt_file(src_train, f"{split_path}/src_train.txt")
        handle_data.write_txt_file(src_val, f"{split_path}/src_val.txt")
        handle_data.write_txt_file(src_test, f"{split_path}/src_test.txt")
        handle_data.write_txt_file(tgt_train, f"{split_path}/tgt_train.txt")
        handle_data.write_txt_file(tgt_val, f"{split_path}/tgt_val.txt")
        handle_data.write_txt_file(tgt_test, f"{split_path}/tgt_test.txt")