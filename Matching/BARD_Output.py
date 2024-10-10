import os
from google.cloud import translate_v3 as translate
from bardapi import Bard

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gen-lang-client-0665319608-f767f723e25d.json"


def translate_method_name(method_name, source_language, target_language):
    client = translate.TranslationServiceClient()

    response = client.translate_text(
        contents=[method_name],
        source_language_code=source_language,
        target_language_code=target_language,
    )

    return response.translations[0].translated_text


if __name__ == "__main__":
    method_name = input("Enter the method name: ")
    source_language = input("Enter the source language (e.g., Java, C#, Python): ")
    target_language = input("Enter the desired language (e.g., Java, C#, Python): ")

    translated_method_name = translate_method_name(method_name, source_language, target_language)

    print(f"Translated method name: {translated_method_name}")

    prompt = f"Suggest a similar method name in {target_language} for: {translated_method_name} (without brackets)."

    response = Bard.completion.create(
        engine="gemini-pro",  # BARD engine
        prompt=prompt,
        max_tokens=50
    )

    similar_method_name = response.choices[0].text.strip()

    print(f"Similar method name in {target_language}: {similar_method_name}")
