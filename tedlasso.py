import os
from openai import OpenAI
import pandas as pd
import numpy as np
# import faiss

from dotenv import load_dotenv  # Optional: For loading environment variables from a .env file

# ACTIVATE ENVIRONMENT: source coaching-assistant-env/bin/activate
# Put grit into bin of environment: https://github.com/getgrit/gritql/releases/latest/download/grit-aarch64-apple-darwin.tar.gz
# MIGRATE OPENAI: grit apply openai

# Set OpenAI API Key
# OpenAI.api_key = "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"
load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Load datasets
players_data = pd.read_csv("gpt_example_training_dataset.csv")

def dataframe_to_string(df, max_rows=30):
    """
    Convert a pandas DataFrame to a readable string.
    Limit the number of rows for token efficiency.
    """
    return df.head(max_rows).to_string(index=False)

# Convert player data to string
player_data_snippet = dataframe_to_string(players_data)

data_context = f"""
Players Data:
{player_data_snippet}
"""

# Function to handle dynamic queries
def coaching_assistant_query(prompt, data_context):
    """
    Use OpenAI API to answer questions with dataset context.
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to the following datasets. You are the assistant coach to the Northwestern Men's Soccer Coach."},
            {"role": "system", "content": data_context},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# Interactive session for coaching assistant
if __name__ == "__main__":
    print("Welcome to the Interactive Coaching Assistant!")
    print("Ask a question about the players data, or type 'exit' to quit.\n")

    while True:
        # Get user input
        prompt = input("Your question: ")
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
        
        # Query assistant with user prompt
        response = coaching_assistant_query(prompt, data_context)
        print("\nAssistant Response:\n")
        print(response)
        print("\n")


'''
# Test the system with a prompt
if __name__ == "__main__":
    # Example prompt
    prompt = "List all the players on the team." # "Which defenders consistently win the highest percentage of defensive duels?"
    response = coaching_assistant_query(prompt, data_context)
    print(response)
'''



'''
ORIGINAL
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message.content)
'''

'''
def coaching_assistant_query(prompt):
    """
    Query OpenAI ChatCompletion API and return the assistant's response.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Or use "gpt-3.5-turbo" if "gpt-4" isn't available
            messages=[
                {"role": "system", "content": "You are a coaching assistant speaking like Ted Lasso."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except openai.error.OpenAIError as e:
        return f"An error occurred: {e}"

# Test function
if __name__ == "__main__":
    prompt = "Tell me about Northwestern players in 2024."
    print(coaching_assistant_query(prompt))
'''
