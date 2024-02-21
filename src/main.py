'''
For handling the pipeline for entire translation framework
'''

from handle_data import preprocess


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

        # preprocess the data
