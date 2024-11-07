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
# from tqdm import tqdm
# Apply nest_asyncio to allow nested event loops
from tqdm.asyncio import tqdm_asyncio

nest_asyncio.apply()

async def process_single_url(browser, url, filename, save_folder, timeout):
    """Process a single URL and save its HTML content"""
    save_path = os.path.join(save_folder, filename)
    try:
        page = await browser.new_page()
        await page.goto(url)
        await asyncio.sleep(timeout)
        html_content = await page.content()
        
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(html_content)
            
        await page.close()
        
        if DEBUG:
            print(f"HTML content saved successfully to: {save_path[-15:]}...")
        return True
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return False

async def download_webpage_html(urls, filenames, save_folder="./documents/", timeout=0.1, max_concurrent=20):
    """
    Download multiple webpages concurrently and save their HTML content
    
    Args:
        urls (list): List of URLs to download
        filenames (list): List of filenames to save the content to
        save_folder (str): Directory to save the files
        timeout (float): Time to wait after page load
        max_concurrent (int): Maximum number of concurrent downloads
    """
    try:
        # Ensure the save_folder exists
        os.makedirs(save_folder, exist_ok=True)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            # Process URLs in batches to control concurrency
            for i in range(0, len(urls), max_concurrent):
                batch_urls = urls[i:i + max_concurrent]
                batch_filenames = filenames[i:i + max_concurrent]
                
                # Create tasks for the current batch
                tasks = [
                    process_single_url(browser, url, filename, save_folder, timeout)
                    for url, filename in zip(batch_urls, batch_filenames)
                ]
                
                # Process batch concurrently with progress bar
                await tqdm_asyncio.gather(
                    *tasks,
                    desc=f"Batch {i//max_concurrent + 1}/{(len(urls) + max_concurrent - 1)//max_concurrent}"
                )
            
            await browser.close()
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

