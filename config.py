import os
from dotenv import load_dotenv

# Try to load from .env file first (for local development)
load_dotenv()

# Get OpenAI API key - will work for both local (.env) and Litstreams (environment variables)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Verify API key is available
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found! Please set it in your .env file for local development "
        "or in your Litstreams environment variables."
    ) 
