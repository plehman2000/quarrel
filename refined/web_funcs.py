import requests
from urllib.parse import urlparse
import os
import html_text
import uuid
import requests

DEBUG = False


def extract_text_from_html_file(file_path, guess_layout=True):
    print(f"FILEPATH: {file_path}")
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Open and read the HTML file
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        # Extract text from the HTML content
        extracted_text = html_text.extract_text(html_content, guess_layout=guess_layout)
        print(f"EXTRACTED TEXT:{extracted_text}")
        return extracted_text

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return None


import requests


def download_webpage_html(
    urls,
    filenames,
    save_folder="./documents/",
    timeout=5.0,
    max_concurrent=10,
    retry_attempts=3,
    api_url="http://localhost:8000",
):
    payload = {
        "urls": urls,
        "filenames": filenames,
        "save_folder": save_folder,
        "timeout": timeout,
        "max_concurrent": max_concurrent,
        "retry_attempts": retry_attempts,
    }

    response = requests.post(f"{api_url}/download/", json=payload)

    if response.status_code == 200:
        return True
    else:
        print(f"API CALL FAILED: {response}")
        return False


# # Example usage of the client
# urls = ["https://example.com", "https://example.org"]
# filenames = ["example1.html", "example2.html"]

# try:
#     saved_files = download_webpages(urls=urls, filenames=filenames)
#     print("Successfully downloaded files:", saved_files)
# except Exception as e:
#     print(f"Error: {e}")


import hashlib
import re


def url_to_unique_name(url: str) -> str:
    # Generate hash
    unique_hash = str(hashlib.sha256(url.encode()).hexdigest())
    # Create filename, ensuring it starts with L and a word (not underscore)
    unique_name = f"L{unique_hash[:25]}_L.html"

    return unique_name
