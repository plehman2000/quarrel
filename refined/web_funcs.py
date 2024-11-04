import requests
from urllib.parse import urlparse
import os
import html_text
import uuid
import requests

DEBUG = False


def extract_text_from_html_file(file_path, guess_layout=True):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Open and read the HTML file
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        # Extract text from the HTML content
        extracted_text = html_text.extract_text(html_content, guess_layout=guess_layout)

        return extracted_text

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return None




# def download_webpage_html(url, filename, save_folder="./documents/", timeout=2):
#     try:

#         #TODO should replace with https://ollama.com/library/reader-lm
# #         filename = filename + ".html"
#         # Ensure the save_folder exists
#         os.makedirs(save_folder, exist_ok=True)
        
#         # Construct the full path to the save file
#         save_path = os.path.join(save_folder, filename)
        
#         # Send a GET request to the URL
#         response = requests.get(url, timeout=timeout)
#         response.raise_for_status()  # Raise an exception for bad status codes
        
#         # Get the HTML content
#         html_content = response.text
        
#         # Save the HTML content to a file
#         with open(save_path, "w", encoding="utf-8") as file:
#             file.write(html_content)
        
#         print(f"HTML content saved successfully to: {save_path[-15:]}...")
#         return save_path

#     except RequestException as e:
#         print(f"An error occurred while accessing the webpage: {e}")
#         return None
#     except IOError as e:
#         print(f"An error occurred while saving the file: {e}")
#         return None
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return None


import os
import asyncio
import nest_asyncio
from playwright.async_api import async_playwright
from tqdm import tqdm
# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()
async def download_webpage_html(urls, filenames, save_folder="./documents/", timeout=0.1):
    try:
        # Ensure the save_folder exists
        os.makedirs(save_folder, exist_ok=True)
        
        # Construct the full path to the save file
        
        # Use Playwright to get the HTML content
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            for url, filename in tqdm(zip(urls, filenames), total=len(urls)):
                try:
                    save_path = os.path.join(save_folder, filename)
                    page = await browser.new_page()
                    await page.goto(url)#, wait_until="domcontentloaded")
                    await asyncio.sleep(timeout)  # Wait for the specified timeout
                    html_content = await page.content()
                    # Save the HTML content to a file
                    with open(save_path, "w", encoding="utf-8") as file:
                        file.write(html_content)
                except Exception:
                    print("potential playwright error/loading a page error, check web_funcs")
            await browser.close()
        

        if DEBUG:
            print(f"HTML content saved successfully to: {save_path[-15:]}...")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

