'''
For handling the pipeline for entire translation framework
'''

import handle_data


if __name__ == '__main__':

    domains = ['dblp/toy.dblp.v12.json']

    for domain in domains:

        # the paths here consider that the main.py is run from its native folder 'src'
        # so 'data' folder will be one up like '../data'
        # filepath : the path of the file
        # path : the path of the directory
        teamsvecs_filepath = f'../data/raw/{domain}/teamsvecs.pkl'
        preprocessed_path = f'../data/preprocessed/{domain}'
        preprocesed_source_filepath = f'{preprocessed_path}/src.txt' # the train-test-val gets appended later
        preprocesed_target_filepath = f'{preprocessed_path}/tgt.txt'

        # preprocess the data to build translation corpus in terms of skills and members
        src, tgt = handle_data.preprocess(teamsvecs_filepath)

        # write the arrays into files
        handle_data.write_txt_file(src, preprocesed_source_filepath)
        handle_data.write_txt_file(tgt, preprocesed_target_filepath)