import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# import fasttext.util
from gensim.models import FastText

fasttext_model_path = 'cc.en.300.bin'  # Replace with the path to your FastText model
ft_model = FastText.load_fasttext_format(fasttext_model_path)

# Load CSV Data (Replace with your CSV file paths)
csv_file_1 = input("Enter the first CSV file name: ")
csv_file_2 = input("Enter the second CSV file name: ")
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

# Function to find the best match and similarity
def find_best_match(description, df):
    if pd.notna(description) and description in ft_model.wv:
        vec_input = ft_model.wv[description]
    else:
        vec_input = None

    best_match = None
    best_similarity = -1

    for index, row in df.iterrows():
        description_csv = row['Description']
        if pd.notna(description_csv) and description_csv in ft_model.wv:
            vec_csv = ft_model.wv[description_csv]
        else:
            vec_csv = None

        if vec_input is not None and vec_csv is not None:
            # Calculate cosine similarity
            similarity = cosine_similarity([vec_input], [vec_csv])[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = row['Method Name']

    return best_match, best_similarity

# Input part
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