import csv
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model_name = "paraphrase-TinyBERT-L6-v2"

model = SentenceTransformer(model_name)

def vectorize_description(description, model):
    description_vector = model.encode([description], convert_to_tensor=True)
    return description_vector[0].cpu().numpy()

def find_best_match(input_description, model, other_csv_file):
    best_match = None
    best_similarity = -1
    description_vector = vectorize_description(input_description, model)

    with open(other_csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row if it exists
        for row in csv_reader:
            if len(row) >= 2:
                other_description = row[1]
                other_description_vector = vectorize_description(other_description, model)
                if description_vector is not None and other_description_vector is not None:
                    similarity_score = util.pytorch_cos_sim(
                        description_vector,
                        other_description_vector
                    )[0][0].item()
                    if similarity_score > best_similarity:
                        best_similarity = similarity_score
                        best_match = row[0]

    return best_match, best_similarity

if __name__ == "__main__":
    csv_file_1 = input("Enter the first CSV file name: ")
    df1 = pd.read_csv(csv_file_1)
    csv_file_2 = input("Enter the second CSV file name: ")

    while True:
        method_name = input("Enter the Method Name to search: ")

        result1 = df1[df1["Method Name"] == method_name]

        if not result1.empty:
            input_description = result1["Description"].values[0]
            print(f"{input_description}")
        else:
            input_description = None
            print("Method Name not found in the 1st CSV file.")

        best_match_2, best_similarity_2 = find_best_match(input_description, model, csv_file_2)

        print(f"Best Match: {best_match_2} with Similarity: {best_similarity_2}\n")
