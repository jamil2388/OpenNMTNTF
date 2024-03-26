import argparse
from sklearn.metrics import roc_auc_score, ndcg_score
import numpy as np
from tqdm import tqdm
import pytrec_eval
import pandas as pd
from scipy.sparse import lil_matrix

# Function to create one-hot vectors
def create_one_hot(line, total_experts):
    one_hot = np.zeros(total_experts)
    indices = [int(expert[1:]) - 1 for expert in line.split()]
    for index in indices:
        one_hot[index] = 1
    return one_hot

def count_total_experts(vocab_file):
    max_digit = 0
    with open(vocab_file, 'r') as file:
        for line in file:
            expert, _ = line.split()
            expert_digit = int(expert[1:])
            if expert_digit > max_digit:
                max_digit = expert_digit
    return max_digit

def calculate_metrics(roc_auc, Y, Y_, metrics={'P_2,5,10', 'recall_2,5,10', 'ndcg_cut_2,5,10', 'map_cut_2,5,10'}):
    qrel = dict();
    run = dict()
    print(f'Building pytrec_eval input for {Y.shape[0]} instances ...')
    from tqdm import tqdm
    with tqdm(total=Y.shape[0]) as pbar:
        for i, (y, y_) in enumerate(zip(Y, Y_)):
            qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in y.nonzero()[1]}
            run['q' + str(i)] = {'d' + str(j): v for j, v in enumerate(y_)}
            pbar.update(1)
    print(f'Evaluating {metrics} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics).evaluate(run))
    df_mean = df.mean(axis=1).to_frame('mean')
    df_mean.loc['aucroc'] = roc_auc # add the rocauc score in the dataframe
    return df_mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ROC AUC Score")
    parser.add_argument("-vocab_tgt", type=str, help="Vocabulary file containing unique members")
    parser.add_argument("-tgt_file", type=str, help="Target file")
    parser.add_argument("-pred_file", type=str, help="Predicted file")
    args = parser.parse_args()

    total_experts = count_total_experts(args.vocab_tgt)

    with open(args.tgt_file, 'r') as file:
        tgt_lines = file.readlines()

    with open(args.pred_file, 'r') as file:
        pred_lines = file.readlines()

    tgt_one_hot_vectors = []
    pred_one_hot_vectors = []

    # Create one-hot vectors for each line
    for i, (tgt_line, pred_line) in tqdm(enumerate(zip(tgt_lines, pred_lines))):
        tgt_one_hot = create_one_hot(tgt_line.strip(), total_experts)
        pred_one_hot = create_one_hot(pred_line.strip(), total_experts)
        tgt_one_hot_vectors.append(tgt_one_hot)
        pred_one_hot_vectors.append(pred_one_hot)

    # Flatten the one-hot vectors
    tgt_one_hot_vectors_flattened = np.array(tgt_one_hot_vectors).flatten()
    pred_one_hot_vectors_flattened = np.array(pred_one_hot_vectors).flatten()

    # # Calculate ROC AUC score
    roc_auc1 = roc_auc_score(tgt_one_hot_vectors_flattened, pred_one_hot_vectors_flattened)
    # roc_auc2 = roc_auc_score(tgt_one_hot_vectors_flattened, pred_one_hot_vectors_flattened, average='micro', multi_class="ovr")
    print("ROC AUC Score 1 : ", roc_auc1)
    # print("ROC AUC Score 2 : ", roc_auc2)

    ## pytrec_eval section
    metrics = calculate_metrics(roc_auc1, lil_matrix(tgt_one_hot_vectors), np.array(pred_one_hot_vectors))
    print(metrics)
    metrics.to_csv(f'{args.pred_file[:-4]}.csv', float_format='%.15f')
