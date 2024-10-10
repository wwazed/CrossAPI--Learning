import csv
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to your downloaded BERT checkpoint files and the corresponding configuration JSON.
model_checkpoint_path = "pytorch_model.bin"  # Replace with the actual path to "pytorch_model.bin"
model_config_path = "tokenizer_config.json"  # Replace with the actual path to the JSON file

# Load the BERT model and tokenizer using the checkpoint and configuration.
model_checkpoint = "bert-base-uncased"
model = BertModel.from_pretrained(model_checkpoint)
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)

# Move the model to GPU
model.to(device)

def vectorize_description(description, tokenizer, model):
    """Vectorizes a description using the BERT model."""
    inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    description_vector = outputs.last_hidden_state.mean(dim=1).squeeze()
    return description_vector.cpu().numpy()  # Convert to NumPy array on CPU

def find_best_match(description, tokenizer, model, desc_data):
    """Finds the best match for a given description in a list of other descriptions using the BERT model."""
    best_match = None
    best_similarity = -1
    description_vector = vectorize_description(description, tokenizer, model)

    for method_name, other_description in desc_data.items():
        other_description_vector = vectorize_description(other_description, tokenizer, model)
        if description_vector is not None and other_description_vector is not None:
            similarity = cosine_similarity([description_vector], [other_description_vector])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = method_name

    return best_match, best_similarity

if __name__ == "__main__":
    # Load the CSV files and store Method Names and Descriptions in a dictionary.
    csv_file_1 = open("JDK.csv", "r", encoding="utf-8")
    csv_reader_1 = csv.reader(csv_file_1)
    desc_data_1 = {row[0]: row[1] for row in csv_reader_1}
    csv_file_1.close()

    csv_file_2 = open("Python.csv", "r", encoding="utf-8")
    csv_reader_2 = csv.reader(csv_file_2)
    desc_data_2 = {row[0]: row[1] for row in csv_reader_2}
    csv_file_2.close()

    while True:
        input_description = input("Enter a description (or 'q' to quit): ")
        if input_description.lower() == 'q':
            break

       # best_match_1, best_similarity_1 = find_best_match(input_description, tokenizer, model, desc_data_1)
        best_match_2, best_similarity_2 = find_best_match(input_description, tokenizer, model, desc_data_2)

      #  print(f"Best Match in CSV File 1 (Method Name): {best_match_1} with Similarity: {best_similarity_1}")
        print(f"Best Match in CSV File 2 (Method Name): {best_match_2} with Similarity: {best_similarity_2}\n")