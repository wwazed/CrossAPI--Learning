import gensim
import pandas as pd
import numpy as np

glove_file = 'glove.840B.300d.txt'

glove_model = gensim.models.KeyedVectors.load_word2vec_format(glove_file, binary=False)


def cosine_similarity_vec(v1, v2):
    return v1.dot(v2) / (v1.norm() * v2.norm())


csv_file_1 = input("Enter the first CSV file name: ")
csv_file_2 = input("Enter the second CSV file name: ")

try:
    df1 = pd.read_csv(csv_file_1)
    df2 = pd.read_csv(csv_file_2)
except FileNotFoundError:
    print("One or both of the CSV files do not exist.")
    exit()

method_name = input("Enter the Method Name to search in the first CSV file: ")

result1 = df1[df1["Method Name"] == method_name]

if not result1.empty:
    input_description = result1["Description"].values[0]
else:
    input_description = None
    print("Method Name not found in the first CSV file.")

input_tokens = input_description.lower().split()
vec_input = np.mean([glove_model[word] for word in input_tokens if word in glove_model], axis=0)

best_match_2 = None
best_similarity_2 = -1

for index, row in df2.iterrows():
    description2 = row['Description']

    tokens2 = description2.lower().split()

    vec_description2 = np.mean([glove_model[word] for word in tokens2 if word in glove_model], axis=0)

    similarity = cosine_similarity_vec(vec_input, vec_description2)

    if similarity > best_similarity_2:
        best_similarity_2 = similarity
        best_match_2 = row['Method Name']

print(f"Best Match in {csv_file_2}: {best_match_2} with Similarity: {best_similarity_2}")