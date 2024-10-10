import pandas as pd
import openai

api_key = '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'

# Specify input and output file paths
input_file = 'JDK_Extra.csv'
output_file = 'output.csv'

openai.api_key = api_key

df = pd.read_csv(input_file)

# Initialize empty lists to store method names and generated descriptions
method_names = []
generated_descriptions = []

# Iterate through method names and generate concise descriptions
for method_name in df['Method Name']:
    prompt = f"Generate a concise 25-word description for the method: '{method_name}'. Write only the main point with general English."
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=25,  # Limit to 25 tokens (words)
        n=1
    )

    generated_description = response.choices[0].text.strip()

    # Append the method name and generated description to their respective lists
    method_names.append(method_name)
    generated_descriptions.append(generated_description)

# Create a DataFrame with method names and generated descriptions and save it to the output CSV file
output_df = pd.DataFrame({'Method Name': method_names, 'Description': generated_descriptions})
output_df.to_csv(output_file, index=False)

print("Concise 25-word descriptions generated and saved to output.csv")
