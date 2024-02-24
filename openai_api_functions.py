import openai

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv() #te
a = 0

GPT_KEY = os.getenv('GPT_KEY')


def call_gpt(user_prompt, system_prompt="Be accurate in your responses.", conversation=[], model="gpt-3.5-turbo",
             force_json=False):
    client = openai.OpenAI(api_key=GPT_KEY)
    user_message = {"role": "user", "content": user_prompt}
    system_message = {"role": "system", "content": system_prompt},

    response_format = {"type": "text"}
    if force_json:
        response_format = {"type": "json_object"}

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}] +
                 conversation +
                 [user_message],
        response_format=response_format
    )
    conversation.append(user_message)
    conversation.append(completion.choices[0].message)
    print(completion.choices[0].message)
    return completion.choices[0].message.content, conversation


def text_to_speech(input_, model="tts-1", voice="alloy"):
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = openai.audio.speech.create(
        model=model,
        voice=voice,
        input=input_
    )
    response.stream_to_file(speech_file_path)


def create_client():
    return openai.OpenAI(api_key=GPT_KEY)


def create_embedding(query, client):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input="query",
        encoding_format="float"
    )
    return response.data[0].embedding
