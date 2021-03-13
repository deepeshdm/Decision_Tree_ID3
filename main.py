
# MUST FOLLOW RULES:
# 1] The last column should contain the class labels.
# 2] Last column should have the name as "label".
# 3] No missing values allowed.

from pprint import pprint
import numpy as np
import random
import pandas as pd

df = pd.read_csv("Iris_Dataset.csv")
# renaming last column
df = df.rename(columns={"species": "label"})


# --------------------------------------------------------------------------------

# Determines if values in columns are continuous / categorical.
def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_threshold = 15

    for column in df.columns:
        unique_values = df[column].unique()
        example_value = unique_values[0]

        if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_threshold):
            feature_types.append("categorical")
        else:
            feature_types.append("continuous")

    return feature_types


# A list containing feature type of each column.
FEATURE_TYPES = determine_type_of_feature(df)


# Method for data splitting
def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    random.seed(0)
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


# checks purity of data i.e if the provided data contain
# more than one class values in its target value column.
# Takes 2D numpy array as parameter.
def check_purity(data):
    label_column = data[:, -1]  # Gets all values from target value column
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False


# Takes data as 2D numpy array & return the class value that appears the most.
def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = \
        np.unique(label_column, return_counts=True)

    LargestCountIndex = counts_unique_classes.argmax()
    classification = unique_classes[LargestCountIndex]
    return classification


# Takes data in 2D numpy array & returns potential split values for all columns.
def get_potential_splits(data):
    potential_splits = {}  # Key is Column number,value is a list of potential splits.
    _, n_columns = data.shape

    for column_index in range(n_columns - 1):
        values = data[:, column_index]  # Gets all values of a column.
        unique_values = np.unique(values)

        potential_splits[column_index] = unique_values

    return potential_splits


# Takes 2D data,split column index & the value at which to split data.
def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]

    return data_below, data_above


# Calculates the "Entropy" pf given 2D data.
def calculate_entropy(data):
    label_column = data[:, -1]
    unique_labels, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * - np.log2(probabilities))

    return entropy


# Calculates "Overall Entropy"
def calculate_overall_entropy(data_below, data_above):
    n_data_points = len(data_below) + len(data_above)

    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points

    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))

    return overall_entropy


# Calculates the column & split value with least entropy.
def determine_best_split(data, potential_splits):
    global best_split_column, best_split_value
    overall_entropy = 999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


# MAIN ALGORITHM.
# Takes data as dataframe or numpy array.
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    global COLUMN_HEADERS, FEATURE_TYPES
    if counter == 0:
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df

    # Base case for recursion.
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        return classification
    else:
        counter += 1
        potential_splits = get_potential_splits(data)

        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification

        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]

        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
        else:
            question = "{} = {}".format(feature_name, split_value)

        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


# Takes single row dataframe and classifies it.
def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


def calculate_accuracy(df, tree):
    # Creating new columns in dataframe.
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))

    # compare column "classification" with "label" and create a new boolean column.
    df["classification_correct"] = df.classification == df.label
    accuracy = df.classification_correct.mean()
    return accuracy


# --------------------------------------------------------------------------------

train_df, test_df = train_test_split(df, 0.2)
tree = decision_tree_algorithm(train_df)

# taking single row from dataframe
example_value = train_df.iloc[0]
# predicting the target class
predicted_value = classify_example(example_value, tree)
print(predicted_value)

# printing accuracy of the model
print(calculate_accuracy(test_df, tree))

