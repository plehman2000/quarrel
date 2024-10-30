from llm_funcs import get_llm_response, extract_info_json


prompt = """Write a cold linkedin message to a tech recruiter for adobe. You are inquring about your applications to the
machine learning engineer role and want to know if theres anything you can do to improve your application, and know when we can expect a decision to be made. Be succint, 3 sentencers or less.
"""
res = get_llm_response(prompt)
print(res)