import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "jondurbin/airoboros-gpt-3.5-turbo-100k-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


def vectorize_description(description, tokenizer, model):
    inputs = tokenizer(description, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embeddings from the output
    description_vector = outputs['logits'].mean(dim=1).squeeze()

    if 'past_key_values' in outputs:
        del outputs['past_key_values']  # Remove past_key_values to avoid the AttributeError

    return description_vector.cpu().numpy()


def find_best_match(input_description, model, other_csv_file):
    best_match = None
    best_similarity = -1
    description_vector = vectorize_description(input_description, tokenizer, model)

    with open(other_csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row if it exists
        for row in csv_reader:
            if len(row) >= 2:
                other_description = row[1]
                other_description_vector = vectorize_description(other_description, tokenizer, model)
                if description_vector is not None and other_description_vector is not None:
                    similarity_score = torch.nn.functional.cosine_similarity(
                        torch.tensor([description_vector], device=device),
                        torch.tensor([other_description_vector], device=device)
                    ).item()
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