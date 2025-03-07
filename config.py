# import os
import os
from dotenv import load_dotenv

# Load variables from a .env file if it exists
load_dotenv()

# Get environment variables; use defaults if necessary
API_KEY = os.getenv("API_KEY")
if API_KEY is None:
    raise ValueError("API_KEY environment variable not set.")

ENDPOINT_URL = os.getenv("ENDPOINT_URL", "https://api.us.inc/hanooman/router/v1/chat/completions")
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "1.42"))
