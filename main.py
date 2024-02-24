import database.database as database
import openai_api_functions as ai

def run():
    ai.call_gpt("hello")


if __name__ == "__main__":
    run()