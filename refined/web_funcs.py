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


########################################################################################################
#Asynchronous Version
########################################################################################################
# import os
# import asyncio
# import nest_asyncio
# from playwright.async_api import async_playwright
# # from tqdm import tqdm
# # Apply nest_asyncio to allow nested event loops
# from tqdm.asyncio import tqdm_asyncio
# from tqdm import tqdm
# # nest_asyncio.apply()

# async def process_single_url(browser, url, filename, save_folder, timeout):
#     import asyncio

#     # loop = asyncio.ProactorEventLoop()
#     loop = asyncio.get_event_loop()

#     asyncio.set_event_loop(loop)

#     """Process a single URL and save its HTML content"""
#     save_path = os.path.join(save_folder, filename)
#     try:
#         page = await browser.new_page()
#         await page.goto(url)
#         await asyncio.sleep(timeout)
#         html_content = await page.content()
        
#         with open(save_path, "w", encoding="utf-8") as file:
#             file.write(html_content)
            
#         await page.close()
        
#         if DEBUG:
#             print(f"HTML content saved successfully to: {save_path[-15:]}...")
#         return True
#     except Exception as e:
#         print(f"Error processing {url}: {str(e)}")
#         return False
 

# async def download_webpage_html(urls, filenames, save_folder="./documents/", timeout=0.1, max_concurrent=20):
#     import asyncio

#     # loop = asyncio.ProactorEventLoop()
#     loop = asyncio.new_event_loop()

#     asyncio.set_event_loop(loop)
#     """
#     Download multiple webpages concurrently and save their HTML content
    
#     Args:
#         urls (list): List of URLs to download
#         filenames (list): List of filenames to save the content to
#         save_folder (str): Directory to save the files
#         timeout (float): Time to wait after page load
#         max_concurrent (int): Maximum number of concurrent downloads
#     """
#     try:
#         # Ensure the save_folder exists
#         os.makedirs(save_folder, exist_ok=True)
#         async with async_playwright() as p:

#             browser = await p.chromium.launch(headless=True)
#             # Process URLs in batches to control concurrency
#             for i, url in tqdm_asyncio(enumerate(urls), total=len(urls)):
#                 # await process_single_url(browser, url, filenames[i], save_folder, timeout)
#                 save_path = os.path.join(save_folder, filenames[i])
#                 try:
#                     page = await browser.new_page()
#                     await page.goto(url)
#                     # await asyncio.sleep(timeout)
#                     html_content = await page.content()

#                     with open(save_path, "w", encoding="utf-8") as file:
#                         file.write(html_content)

#                     await page.close()

#                     if DEBUG:
#                         print(f"HTML content saved successfully to: {save_path[-15:]}...")
#                 except Exception as e:
#                     print(f"Error processing {url}: {str(e)}")
    




#             # for i in range(0, len(urls), max_concurrent):
#             #     batch_urls = urls[i:i + max_concurrent]
#             #     batch_filenames = filenames[i:i + max_concurrent]
#             #     # Create tasks for the current batch
#             #     tasks = [
#             #         process_single_url(browser, url, filename, save_folder, timeout)
#             #         for url, filename in zip(batch_urls, batch_filenames)
#             #     ]
#             #     # Process batch concurrently with progress bar
#             #     await tqdm_asyncio.gather(
#             #         *tasks,
#             #         desc=f"Batch {i//max_concurrent + 1}/{(len(urls) + max_concurrent - 1)//max_concurrent}"
#             #     )
#             await browser.close()
#         return True    
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return False



# asyncio.run(download_webpage_html(["https://example.com/"], ["n.html"]))
########################################################################################################
#Synchronous Version
########################################################################################################
# from playwright.sync_api import sync_playwright
# import os
# from tqdm import tqdm
# import nest_asyncio
# def download_webpage_html(urls, filenames, save_folder="./documents/", timeout=0.1, max_concurrent=20):
#     """
#     Download multiple webpages and save their HTML content synchronously
    
#     Args:
#         urls (list): List of URLs to download
#         filenames (list): List of filenames to save the content to
#         save_folder (str): Directory to save the files
#         timeout (float): Time to wait after page load
#         max_concurrent (int): Maximum number of concurrent downloads
        
#     Returns:
#         bool: True if successful, False if an error occurred
#     """
#     nest_asyncio.apply()
#     try:
#         # Ensure the save_folder exists
#         os.makedirs(save_folder, exist_ok=True)
        
#         with sync_playwright() as p:
#             browser = p.chromium.launch(headless=True)
            
#             # Process URLs in batches
#             for i in range(0, len(urls), max_concurrent):
#                 batch_urls = urls[i:i + max_concurrent]
#                 batch_filenames = filenames[i:i + max_concurrent]
                
#                 # Process each URL in the current batch
#                 for url, filename in tqdm(
#                     zip(batch_urls, batch_filenames),
#                     desc=f"Batch {i//max_concurrent + 1}/{(len(urls) + max_concurrent - 1)//max_concurrent}"
#                 ):
#                     save_path = os.path.join(save_folder, filename)
#                     try:
#                         page = browser.new_page()
#                         page.goto(url)
#                         page.wait_for_timeout(timeout * 1000)  # Convert to milliseconds
#                         html_content = page.content()
                        
#                         with open(save_path, "w", encoding="utf-8") as file:
#                             file.write(html_content)
                            
#                         page.close()
                        
#                         if DEBUG:
#                             print(f"HTML content saved successfully to: {save_path[-15:]}...")
                            
#                     except Exception as e:
#                         print(f"Error processing {url}: {str(e)}")
#                         continue
                        
#             browser.close()
#         return True
        
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return False
    


# download_webpage_html(["https://example.com/"], ["n.html"])


#################################



#############################################
#############################################
#My Version 
#############################################
#############################################



# from tqdm import tqdm
# def download_webpage_html(
#     urls: Union[str, List[str]], 
#     filenames: Union[str, List[str]], 
#     save_folder: str = "./documents/", 
#     timeout: float = 0.1,
#     max_concurrent: int = 20
# ) -> List[str]:
#     """
#     Downloads HTML content from specified URLs using BeautifulSoup and saves them to local files.
    
#     Args:
#         urls: Single URL string or list of URLs to download
#         filenames: Single filename string or list of filenames to save as
#         save_folder: Directory to save the downloaded files (default: "./documents/")
#         timeout: Time to wait between downloads (default: 0.1s)
#         max_concurrent: Maximum number of concurrent downloads (default: 20)
        
#     Returns:
#         List of paths to the saved files
    
#     Raises:
#         ValueError: If lengths of urls and filenames don't match
#         URLError: If there's an error downloading the webpage
#         HTTPError: If there's an HTTP error during download
#     """
#     # Convert single strings to lists for consistent processing
#     if isinstance(urls, str):
#         urls = [urls]
#     if isinstance(filenames, str):
#         filenames = [filenames]
        
#     # Validate inputs
#     if len(urls) != len(filenames):
#         raise ValueError("Number of URLs must match number of filenames")
    
#     saved_files = []
    
#     # Process URLs in batches of max_concurrent
#     for url, filename in tqdm(zip(urls, filenames), total=len(urls)):
#         try:
#             # Create downloader instance
#             downloader = WebPageDownloader(url, save_folder)
            
#             # Ensure directory exists
#             downloader.make_dir(save_folder)
            
#             # Save the webpage
#             saved_file = downloader.save_page(filename)
#             saved_files.append(saved_file)
            
#             # Sleep to avoid overwhelming the server
#             time.sleep(timeout)
            
#         except (HTTPError, URLError) as e:
#             print(f"Error downloading {url}: {str(e)}")
#         except Exception as e:
#             print(f"Unexpected error processing {url}: {str(e)}")
                
#     return saved_files

    







#############################################
#############################################
#CLAUDE VERSION
#############################################
#############################################
# import asyncio
# import aiohttp
# import os
# from typing import Union, List
# from urllib.error import URLError, HTTPError
# from bs4 import BeautifulSoup
# from tqdm import tqdm
# import logging

# class WebPageDownloader:
#     def __init__(self, session: aiohttp.ClientSession, save_folder: str):
#         self.session = session
#         self.save_folder = save_folder

#     async def download_page(self, url: str, timeout: float) -> str:
#         async with self.session.get(url, timeout=timeout) as response:
#             response.raise_for_status()
#             return await response.text()

#     def save_content(self, content: str, filename: str) -> str:
#         filepath = os.path.join(self.save_folder, filename)
#         with open(filepath, 'w', encoding='utf-8') as f:
#             f.write(content)
#         return filepath

# async def download_webpage_html(
#     urls: Union[str, List[str]],
#     filenames: Union[str, List[str]],
#     save_folder: str = "./documents/",
#     timeout: float = 30.0,  # Increased default timeout
#     max_concurrent: int = 20,
#     retry_attempts: int = 3,
#     retry_delay: float = 1.0
# ) -> List[str]:
#     """
#     Downloads HTML content from specified URLs using aiohttp and saves them to local files.
    
#     Args:
#         urls: Single URL string or list of URLs to download
#         filenames: Single filename string or list of filenames to save as
#         save_folder: Directory to save the downloaded files (default: "./documents/")
#         timeout: Request timeout in seconds (default: 30s)
#         max_concurrent: Maximum number of concurrent downloads (default: 20)
#         retry_attempts: Number of retry attempts for failed downloads (default: 3)
#         retry_delay: Delay between retry attempts in seconds (default: 1.0)
    
#     Returns:
#         List of paths to the saved files
    
#     Raises:
#         ValueError: If lengths of urls and filenames don't match
#         aiohttp.ClientError: If there's an error downloading the webpage
#     """
#     # Convert single strings to lists
#     if isinstance(urls, str):
#         urls = [urls]
#     if isinstance(filenames, str):
#         filenames = [filenames]
    
#     # Validate inputs
#     if len(urls) != len(filenames):
#         raise ValueError("Number of URLs must match number of filenames")

#     # Create save directory if it doesn't exist
#     os.makedirs(save_folder, exist_ok=True)

#     # Initialize logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     saved_files = []
    
#     # Create semaphore for concurrent downloads
#     semaphore = asyncio.Semaphore(max_concurrent)

#     async def download_with_retry(url: str, filename: str, downloader: WebPageDownloader) -> str:
#         for attempt in range(retry_attempts):
#             try:
#                 async with semaphore:
#                     content = await downloader.download_page(url, timeout)
#                     filepath = downloader.save_content(content, filename)
#                     logger.info(f"Successfully downloaded {url} to {filepath}")
#                     return filepath
#             except asyncio.TimeoutError:
#                 logger.warning(f"Timeout downloading {url} (attempt {attempt + 1}/{retry_attempts})")
#                 if attempt < retry_attempts - 1:
#                     await asyncio.sleep(retry_delay)
#             except Exception as e:
#                 logger.error(f"Error downloading {url} (attempt {attempt + 1}/{retry_attempts}): {str(e)}")
#                 if attempt < retry_attempts - 1:
#                     await asyncio.sleep(retry_delay)
#         return ""

#     async with aiohttp.ClientSession() as session:
#         downloader = WebPageDownloader(session, save_folder)
        
#         # Create download tasks
#         tasks = [
#             download_with_retry(url, filename, downloader)
#             for url, filename in zip(urls, filenames)
#         ]
        
#         # Process downloads with progress bar
#         for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
#             result = await task
#             if result:
#                 saved_files.append(result)

#     return saved_files

# # Example usage
# async def main():
#     urls = ["https://example.com", "https://example.org"]
#     filenames = ["example1.html", "example2.html"]
    
#     saved_files = await download_webpage_html(
#         urls=urls,
#         filenames=filenames,
#         save_folder = "./documents/",
#         timeout=5.0,
#         max_concurrent=5,
#         retry_attempts=3
#     )

# # Run the async function
# if __name__ == "__main__":
#     asyncio.run(main())



import requests

def download_webpage_html(
    urls,
    filenames,
    save_folder="./documents/",
    timeout=5.0,
    max_concurrent=5,
    retry_attempts=3,
    api_url="http://localhost:8000"
):
    payload = {
        "urls": urls,
        "filenames": filenames,
        "save_folder": save_folder,
        "timeout": timeout,
        "max_concurrent": max_concurrent,
        "retry_attempts": retry_attempts
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

