import argparse
from sklearn.metrics import roc_auc_score, ndcg_score
import numpy as np
from tqdm import tqdm

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

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(tgt_one_hot_vectors_flattened, pred_one_hot_vectors_flattened)
    print("ROC AUC Score : ", roc_auc)

    # ndcg
    ndcg = ndcg_score(tgt_one_hot_vectors, pred_one_hot_vectors)
    print("ndcg : ", ndcg)
