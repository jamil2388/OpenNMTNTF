import argparse
import random
import re

total_experts = 14210
# Function to replace a member of the target list with another member
def replace_member(target_list):
    # Choose a random member to replace
    index_to_replace = random.randint(0, len(target_list) - 1)
    # Generate a random member to replace with
    new_member = f"m{random.randint(1, total_experts)}"
    # Replace the member
    target_list[index_to_replace] = new_member


# Function to remove a random member from the target list
def remove_member(target_list):
    # Choose a random member to remove
    index_to_remove = random.randint(0, len(target_list) - 1)
    # Remove the member
    del target_list[index_to_remove]

# extract the number of distinct members or skills from the file
# Function to extract unique digits from a line
def extract_unique_digits(line):
    # Find all digits in the line using regular expression
    digits = re.findall(r'\d+', line)
    # Convert the digits to integers and return unique digits using a set
    return set(map(int, digits))


# Function to count total unique digits in the file
def count_total_unique_digits(filename):
    # Initialize an empty set to store unique digits
    unique_digits = set()

    # Open the file and process each line
    with open(filename, 'r') as file:
        for line in file:
            # Extract unique digits from the line and update the set
            unique_digits.update(extract_unique_digits(line))

    # Return the total count of unique digits
    return len(unique_digits)

# experiment line
print(count_total_unique_digits('../data/input/dblp/dblp.v12.json.filtered.mt75.ts3/src.txt'))

# Argument parsing
parser = argparse.ArgumentParser(description="Apply manipulations to target data.")
parser.add_argument("-tgt_file", help="Filename of the target data file")
parser.add_argument("-pred_file", help="Filename of the prediction data file")
parser.add_argument("-w", nargs = '+', type = int, help="Weights for manipulation (replace, remove, none)")
args = parser.parse_args()

# Read the target test data from tgt_file
with open(args.tgt_file, "r") as tgt_file:
    target_data = tgt_file.readlines()

# Apply manipulations
for i in range(len(target_data)):
    # Split the line into individual members
    target_list = target_data[i].strip().split()

    # Randomly choose between replacement, removal, or doing nothing
    manipulation_type = random.choices(["replace", "remove", "none"], weights=args.w, k=1)[0]

    if manipulation_type == "replace":
        replace_member(target_list)
    elif manipulation_type == "remove":
        remove_member(target_list)

    # Update the line in target data if not "none"
    if manipulation_type != "none":
        target_data[i] = " ".join(target_list) + "\n"

# Write the modified target data to pred_file
with open(args.pred_file, "w") as pred_file:
    pred_file.writelines(target_data)

print("Manipulations applied and saved to", args.pred_file)

# Compare tgt_file and pred_file
different_rows = []
with open(args.tgt_file, "r") as tgt_file, open(args.pred_file, "r") as pred_file:
    for line_num, (tgt_line, pred_line) in enumerate(zip(tgt_file, pred_file), start=1):
        if tgt_line.strip() != pred_line.strip():
            different_rows.append(line_num)

print("Rows with differences between", args.tgt_file, "and", args.pred_file, ":", different_rows)
