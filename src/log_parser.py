import re

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as log_file:
        log_content = log_file.read()

    # Define regular expressions to extract relevant information
    train_perplexity_pattern = r'\[.*?\] Train perplexity: (\d+\.\d+)'
    train_accuracy_pattern = r'\[.*?\] Train accuracy: (\d+\.\d+)'
    validation_perplexity_pattern = r'\[.*?\] Validation perplexity: (\d+\.\d+)'
    validation_accuracy_pattern = r'\[.*?\] Validation accuracy: (\d+\.\d+)'

    # Find all matches of the patterns in the log content
    train_perplexity_scores = re.findall(train_perplexity_pattern, log_content)
    train_accuracy_scores = re.findall(train_accuracy_pattern, log_content)
    validation_perplexity_scores = re.findall(validation_perplexity_pattern, log_content)
    validation_accuracy_scores = re.findall(validation_accuracy_pattern, log_content)

    # Print the scores along with numbering for each fold
    num_folds = min(len(train_perplexity_scores), len(validation_perplexity_scores))
    for fold in range(num_folds):
        print(f"Fold {fold + 1}:")
        print(f"  Train accuracy: {train_accuracy_scores[fold]}")
        print(f"  Validation accuracy: {validation_accuracy_scores[fold]}")
        print(f"  Train perplexity: {train_perplexity_scores[fold]}")
        print(f"  Validation perplexity: {validation_perplexity_scores[fold]}")
        print()

log_file_path = '../data/logs/train.e50000.v10000.out'
parse_log_file(log_file_path)
