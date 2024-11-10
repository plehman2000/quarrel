import gradio as gr
import asyncio
from web_funcs import download_webpage_html


test_urls = ["https://example.com", "https://httpbin.org/html"]
test_filenames = ["example.html", "httpbin.html"]
save_folder = "./documents/"


res =  download_webpage_html(urls=test_urls, filenames=test_filenames)

print(res)




async def async_download_websites(prompt):
    download_webpage_html(urls=test_urls, filenames=test_filenames)
    return res

async def download_websites_wrapper(prompt):
    try:
        return await asyncio.wait_for(async_download_websites(prompt), timeout=6)
    except asyncio.TimeoutError:
        return "Processing timed out."

iface = gr.Interface(
    fn=download_websites_wrapper,
    inputs="text",
    outputs="text",
    title="Async Prompt Processor with Timeout",
    description="Enter a prompt to process asynchronously. Times out after 5 seconds.",
    concurrency_limit=10
)

iface.launch()