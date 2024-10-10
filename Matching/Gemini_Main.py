import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = "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"

try:
    # Initialize the DeciLM-6b model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b").to(device)
except Exception as e:
    print("An error occurred while loading the model:", str(e))
    exit()

def vectorize_description(description, tokenizer, model):
    """Vectorizes a description using the DeciLM-6b model."""
    inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    description_vector = outputs.logits.mean(dim=1).squeeze().cpu().numpy()
    return description_vector if description_vector is not None else None

if __name__ == "__main__":
    file1_path = input("Enter the first TSV file name: ")
    file2_path = input("Enter the second TSV file name: ")
    try:
        df1 = pd.read_csv(file1_path, encoding='utf-8', sep='\t')
        df2 = pd.read_csv(file2_path, encoding='utf-8', sep='\t')
    except FileNotFoundError:
        print("One or both of the TSV files do not exist.")
        exit()

    descriptions_1 = df1["Description"].tolist()
    descriptions_2 = df2["Description"].tolist()
    
    all_descriptions = descriptions_1 + descriptions_2
    all_vectors = []

    for description in all_descriptions:
        vector = vectorize_description(description, tokenizer, model)
        if vector is not None:
            all_vectors.append(vector)
        else:
            all_vectors.append(np.zeros(512))  # Assuming the vector size is 512

    similarity_matrix = cosine_similarity(all_vectors)
    correlation_matrix = pd.DataFrame(similarity_matrix, columns=df1["Method Name"].tolist() + df2["Method Name"].tolist(),
                                      index=df1["Method Name"].tolist() + df2["Method Name"].tolist())

    print(correlation_matrix)