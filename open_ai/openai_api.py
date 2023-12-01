import openai
import os

openai.api_key = "YOUR_KEY_HERE"
text = "TEXT_TO_SUMMARIZE_HERE"
messages = [
    {"role": "user", "content": f"Fais un résumé structuré en 7 à 10 lignes des du texte suivant: {text}"}
]
rep = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
print(rep["choices"][0]["message"]["content"])