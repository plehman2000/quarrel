from quart import Quart, request, jsonify
from pydantic import BaseModel, ValidationError
from typing import List
import aiohttp
import asyncio
import os
from pathlib import Path
import dotenv # type: ignore
import pickle

from web_funcs import url_to_unique_name


dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

URL_DICTIONARY_FILEPATH = os.getenv("URL_DICTIONARY_FILEPATH")

app = Quart(__name__)

class DownloadRequest(BaseModel):
    urls: List[str]
    filenames: List[str]
    save_folder: str = "./documents/"
    timeout: float = 5.0
    max_concurrent: int = 5
    retry_attempts: int = 3

async def download_single(
    session: aiohttp.ClientSession,
    url: str,
    filename: str,
    save_folder: str,
    timeout: float,
    retry_attempts: int
) -> str:
    for attempt in range(retry_attempts):
        try:
            await asyncio.sleep(1) 
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    content = await response.text()
                    full_path = os.path.join(save_folder, filename)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    print(f"Saving: {full_path}")
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return full_path
        except asyncio.CancelledError:
            raise Exception(f"Download task for {url} was cancelled")
            return None
        

@app.route("/download/", methods=["POST"])
async def download_webpages():
    try:
        # Parse and validate the JSON request body asynchronously
        json_data = await request.get_json()
        download_request = DownloadRequest(**json_data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    if len(download_request.urls) != len(download_request.filenames):
        return jsonify({"error": "Number of URLs must match number of filenames"}), 400

    # Ensure the save folder exists
    Path(download_request.save_folder).mkdir(parents=True, exist_ok=True)
    saved_files = []
    
    # Use a connector with a limited number of concurrent connections
    connector = aiohttp.TCPConnector(limit=download_request.max_concurrent)
    urls = []
    filenames = []

    site_id_dict = None
    if os.path.exists(URL_DICTIONARY_FILEPATH):
        with open(URL_DICTIONARY_FILEPATH, 'rb') as file:
            site_id_dict = pickle.load(file)
    else:
        with open(URL_DICTIONARY_FILEPATH, 'wb') as file:
            pickle.dump({}, file)


    for u,f in zip(download_request.urls, download_request.filenames):
        if url_to_unique_name(f) not in site_id_dict:
            urls.append(u)
            filenames.append(f)

    print(f"Saved {len(filenames) - len(list(download_request.filenames))} Web Download API Calls...")

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for url, filename in zip(urls, filenames):
                task = download_single(
                    session=session,
                    url=url,
                    filename=filename,
                    save_folder=download_request.save_folder,
                    timeout=download_request.timeout,
                    retry_attempts=download_request.retry_attempts
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    raise result
                if result:
                    saved_files.append(result)
                
            return jsonify(saved_files)
    

    except asyncio.CancelledError:
        return jsonify({"error": "Download operation was cancelled"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app with an ASGI server such as hypercorn
# hypercorn server:app --bind 0.0.0.0:8000

if __name__ == "__main__":
    app.run(debug=True)

"""
hypercorn server:app --bind 0.0.0.0:8000

"""