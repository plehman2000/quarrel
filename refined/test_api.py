from prover import download_webpage_html


test_urls = ["https://example.com", "https://httpbin.org/html"]
test_filenames = ["example.html", "httpbin.html"]
save_folder = "./documents/"


res =  download_webpage_html(urls=test_urls,
filenames=test_filenames
save_folder=save_folder)

print(res)

