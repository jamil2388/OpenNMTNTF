'''
For handling the pipeline for entire translation framework
'''

import handle_data
from sklearn.model_selection import train_test_split
from w2v import W2V

# global variables
random_state = 10

# to streamline all w2v processes and save the embeddings
# also return the filepath of the saved embeddings
# mode = src or tgt
def run_w2v(input_filepath, output_folder, mode):
    w2v = W2V(input_filepath)
    w2v.train()
    print()
    try :
        print(f's9 : {w2v.model.wv.get_vector("s9")}')
        print(f'most similar ro s9 : {w2v.model.wv.most_similar("s9")}')
    except KeyError:
        pass
    return w2v.save_embeddings(output_folder, mode)

if __name__ == '__main__':

    # domains = ['dblp/toy.dblp.v12.json']
    # domains = ['dblp/dblp.v12.json.filtered.mt100.ts5']
    domains = ['dblp/dblp.v12.json.filtered.mt75.ts3']

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

        # save src and tgt corpus for generating pre_trained embeddings
        handle_data.write_txt_file(src, f"{split_path}/src.txt")
        handle_data.write_txt_file(tgt, f"{split_path}/tgt.txt")

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

        run_w2v(preprocesed_source_filepath, f'{preprocessed_path}', mode = 'src')
        run_w2v(preprocesed_target_filepath, f'{preprocessed_path}', mode = 'tgt')



