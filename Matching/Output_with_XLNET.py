import pandas as pd
import torch
from transformers import XLNetTokenizer, XLNetModel
from sklearn.metrics.pairwise import cosine_similarity

model_name = "xlnet-base-cased"
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetModel.from_pretrained(model_name)

csv_file_1 = input("Enter the first CSV file name: ")
csv_file_2 = input("Enter the second CSV file name: ")
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

def generate_embeddings(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        output = model(input_ids)[0].mean(dim=1)

    return output

def calculate_cosine_similarity(vector1, vector2):
    vector1 = vector1.squeeze().numpy()
    vector2 = vector2.squeeze().numpy()
    return cosine_similarity([vector1], [vector2])[0][0]

def find_best_matches(description, df, top_k=20):
    vec_input = generate_embeddings(description, model, tokenizer)

    best_matches = []  # Store multiple best matches

    for index, row in df.iterrows():
        description_csv = row['Description']

        # Check if the description_csv is missing or NaN
        if pd.isna(description_csv):
            continue

        vec_csv = generate_embeddings(description_csv, model, tokenizer)
        similarity = calculate_cosine_similarity(vec_input, vec_csv)
        best_matches.append((row['Method Name'], similarity))

    best_matches.sort(key=lambda x: x[1], reverse=True)

    best_matches = best_matches[:top_k]

    return best_matches

method_name = input("Enter the Method Name to search: ")

# Search for the method name in the first file
result1 = df1[df1["Method Name"] == method_name]

if not result1.empty:
    user_description = result1["Description"].values[0]
    print(f"{user_description}")
else:
    user_description = None
    print("Method Name not found in the 1st CSV file.")

print("\nMatches in CSV File 2:")
matches_file2 = find_best_matches(user_description, df2)
for idx, (method_name, similarity) in enumerate(matches_file2, start=1):
    print(f"{idx}. Method Name: {method_name}, Similarity: {similarity}")