# Import necessary libraries
from openai import OpenAI  # For interacting with OpenAI's API
from transformers import GPT2TokenizerFast  # For tokenizing the input text
import time  # For implementing retries with delay
import random  # For randomizing retry delays

# Initialize the OpenAI API client
client = OpenAI(
    api_key='API_KEY',  # Replace 'API_KEY' with your actual OpenAI API key
)

# Load a tokenizer for counting tokens in the input prompt
tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo')

def num_tokens_from_string(string: str):
    """
    Count the number of tokens in a given string using the tokenizer.
    
    Args:
        string (str): The input string to tokenize.
        
    Returns:
        int: The number of tokens in the input string.
    """
    num_tokens = len(tokenizer.encode(string))
    return num_tokens

def call_api(prompt="Hello", retries=5, temperature=0.0):
    """
    Send a prompt to the OpenAI API with automatic retries for reliability.
    
    Args:
        prompt (str): The input prompt to send to the OpenAI model.
        retries (int): The maximum number of retry attempts for failed API calls.
        temperature (float): Temperature of the model's sampling distribution.  Set temperature to 0.0 for deterministic output.
        
    Returns:
        str: The model's response or a failure message.
    """
    done = False
    retry_count = 0

    # Model selection based on token count
    model_4k = "gpt-4-turbo"          # Default model for prompts < 3800 tokens
    model_16k = "gpt-4-turbo-16k"      # Larger model for prompts >= 3800 tokens

    # Determine token count for the prompt
    num_tokens = num_tokens_from_string(prompt)
    if num_tokens >= 16000:
        return "<<FAILED>>"  # Return if prompt exceeds max token limit
    
    # Select model based on token count (allows for flexibility in model selection)
    model = model_4k if num_tokens < 3800 else model_16k

    # Retry mechanism to handle transient API errors
    while not done:
        try:
            # Call the OpenAI API with the specified prompt and model
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature 
            )
            done = True  # Mark completion if call is successful
            
        except Exception as error:
            # Handle retryable errors with an exponential backoff delay
            if "Please retry after" in str(error):
                timeout = int(str(error).split("Please retry after ")[1].split(" second")[0]) + 5 * random.random()
                print(f"Waiting {timeout:.2f}s before retrying OpenAI API ({error})")
                time.sleep(timeout)
            elif retry_count < retries:
                print(f"Retrying OpenAI API ({retry_count + 1}/{retries}) due to error: {error}")
                time.sleep(2)  # Fixed delay between retries
                retry_count += 1
            else:
                print(f"API call failed after {retries} attempts: {error}")
                return "<<FAILED>>"  # Return failure message if max retries exceeded
    
    # Extract the response content and return it
    reply = response.choices[0].message.content
    return reply
