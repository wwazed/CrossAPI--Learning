import pandas as pd
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import numpy as np

word2vec_model_path = "GoogleNews-vectors-negative300.bin.gz"
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True, limit=500000)

file1_path = input("Enter the first CSV file name: ")
file2_path = input("Enter the second CSV file name: ")

try:
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
except FileNotFoundError:
    print("One or both of the CSV files do not exist.")
    exit()
def preprocess_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        return tokens
    else:
        return []

def cosine_similarity_vec(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

method_name = input("Enter the Method Name to search: ")

# Search for the method name in the first file
result1 = df1[df1["Method Name"] == method_name]

if not result1.empty:
    input_description = result1["Description"].values[0]
else:
    input_description = None
    print("Method Name not found in the 1st CSV file.")

# Initialize lists to store the top 20 similarities and method names
top_matches_2 = []

input_tokens = preprocess_text(input_description)

for index, row in df2.iterrows():
    description = row['Description']
    tokens = preprocess_text(description)

    vec_input = [word2vec_model[word] for word in input_tokens if word in word2vec_model]
    vec_csv = [word2vec_model[word] for word in tokens if word in word2vec_model]

    if vec_input and vec_csv:
        similarity = cosine_similarity_vec(vec_input[0], vec_csv[0])

        # Add the similarity and method name to the list
        top_matches_2.append((similarity, row['Method Name']))

# Sort the list of top matches by similarity in descending order
top_matches_2.sort(key=lambda x: x[0], reverse=True)

# Select the top 20 matches
top_matches_2 = top_matches_2[:20]

print(f"For Input Description: {input_description}")

print(f"Top 20 Matches in {file2_path}:")
for i, (similarity, method_name) in enumerate(top_matches_2):
    print(f"{i+1}. {method_name} with Similarity: {similarity}")