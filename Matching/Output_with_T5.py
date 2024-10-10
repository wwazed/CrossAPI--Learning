import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

csv_file_1 = input("Enter the first CSV file name: ")
csv_file_2 = input("Enter the second CSV file name: ")
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)


def generate_embeddings(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(device)

    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        output = model.encoder(input_ids).last_hidden_state.mean(dim=1)

    return output


def calculate_cosine_similarity(vector1, vector2):
    vector1 = vector1.squeeze().numpy()
    vector2 = vector2.squeeze().numpy()
    return cosine_similarity([vector1], [vector2])[0][0]


def find_best_match(description, other_descriptions):
    vec_input = generate_embeddings(description, model, tokenizer)

    best_match = None
    best_similarity = -1.0

    for index, row in other_descriptions.iterrows():
        description_csv = row['Description']

        if pd.isna(description_csv):
            continue

        vec_csv = generate_embeddings(description_csv, model, tokenizer)

        similarity = calculate_cosine_similarity(vec_input, vec_csv)

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = row['Method Name']

    return best_match


# Take user input for the method name
while True:
    method_name = input("Enter the Method Name to search: ")

    # Search for the method name in the first file
    result1 = df1[df1["Method Name"] == method_name]

    if not result1.empty:
        input_description = result1["Description"].values[0]
        print(f"{input_description}")
    else:
        input_description = None
        print("Method Name not found in the 1st CSV file.")

    best_match_2, best_similarity_2 = find_best_match(input_description, df2)

    print(f"Best Match in CSV File 2: {best_match_2} with Similarity: {best_similarity_2}\n")