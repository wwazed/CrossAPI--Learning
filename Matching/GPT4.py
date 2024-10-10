import openai

# Replace 'YOUR_API_KEY' with your OpenAI GPT-4 API key
api_key = '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'

# Define the GPT-4 API endpoint
openai.api_key = api_key

def find_best_match(user_input, lang_input):
    prompt = f"Find the 20 most similar method name to '{user_input}' from '{lang_IO1}' for the programming language {lang_input}. Return only the method name without any bracket, no extra text."

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=50,  # Adjust the max tokens as needed
        n=1  # Number of completions to generate
    )

    completion_text = response.choices[0].text
    matched_method_name = completion_text.strip()

    return matched_method_name

# Method name and programming language inputs
user_input = input("Enter the Method Name: ")
lang_IO1 = input("Enter the Programming Language: ")
lang_input = input("Enter the Desired Programming Language: ")

# Find the best-matched method name using GPT-4
best_match = find_best_match(user_input, lang_input)

# Display the result
# print(f"The most similar method name for '{user_input}' in the programming language '{lang_input}' is:")
print(f"Method Name: {best_match}")