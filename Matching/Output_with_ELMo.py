import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the locally downloaded ELMo model
elmo = hub.load("https://tfhub.dev/google/elmo/3")

def vectorize_description(description):
    """Vectorizes a description using the ELMo model."""
    tokens = [description]  # Wrap the description in a list
    embeddings = elmo.signatures["default"](tf.constant(tokens))["elmo"]
    description_vector = tf.reduce_mean(embeddings, axis=1).numpy()[0]  # Reduce to a single vector
    return description_vector

def match_descriptions(description, other_descriptions):
    """Matches a description to the most similar description in a list of other descriptions using ELMo."""
    best_match = None
    best_match_score = -np.inf
    description_vector = vectorize_description(description)

    for other_description in other_descriptions:
        other_description_vector = vectorize_description(other_description)
        score = np.dot(description_vector, other_description_vector)
        if score > best_match_score:
            best_match_score = score
            best_match = other_description
    return best_match

def find_top_20_matches(description, other_descriptions):
    top_20_matches = []
    description_vector = vectorize_description(description)

    for other_description in other_descriptions:
        other_description_vector = vectorize_description(other_description)
        score = np.dot(description_vector, other_description_vector)
        top_20_matches.append((other_description, score))

    top_20_matches = sorted(top_20_matches, key=lambda x: x[1], reverse=True)[:20]
    return top_20_matches

if __name__ == "__main__":
    file1_path = input("Enter the first CSV file name: ")
    file2_path = input("Enter the second CSV file name: ")

    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except FileNotFoundError:
        print("One or both of the CSV files do not exist.")
        exit()

    descriptions_1 = df1["Description"].tolist()
    descriptions_2 = df2["Description"].tolist()

    method_name = input("Enter the Method Name to search: ")

    result1 = df1[df1["Method Name"] == method_name]

    if not result1.empty:
        user_description = result1["Description"].values[0]
    else:
        print("Method Name not found in the CSV file.")
        exit()

    top_20_matches_file2 = find_top_20_matches(user_description, descriptions_2)
    best_match_file2 = match_descriptions(user_description, descriptions_2)

    print("\nTop 20 Matches in CSV File 2:")
    for idx, (match, score) in enumerate(top_20_matches_file2, start=1):
        print(f"{idx}. Description: {match}, Similarity Score: {score}")

    print("\nBest Match in CSV File 2:")
    print(f"Description: {best_match_file2}")