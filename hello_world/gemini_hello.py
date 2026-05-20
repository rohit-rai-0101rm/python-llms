import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore")


load_dotenv()

from google import genai


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

try:
   
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="How does AI work? Explain in simple words."
    )

    print("\nResponse:\n")
    print(response.text)

except Exception as e:
    print("\nError occurred:\n")
    print(e)
