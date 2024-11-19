import ollama
import json


# MODEL = "Hudson/llama3.1-uncensored:8b"
# MODEL = 'dolphin-llama3'
MODEL = 'CognitiveComputations/dolphin-llama3.1'
SMALL_MODEL = 'tinydolphin'
BIG_MODEL = "dolphin-llama3:70b"
# MODEL = 'llama3.2'




from openai import OpenAI
from openai import OpenAI
import json
import os
import dotenv # type: ignore

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))



#########################################################################################
#UNCOMMENT ME TO RUN COMPLETELY "OFFLINE" (STILL NEED TO USE WEB SEARCH API)
# Make sure to comment out the duplicate functions below after uncommenting these
#########################################################################################
"""
def get_llm_response(prompt, model = MODEL):
    response = ollama.generate(model=model, prompt=prompt)
    output = response['response']
    return output


def get_llm_json_response(prompt, model=MODEL):
    response = ollama.generate(model=model, prompt=prompt,format='json')
    output = response['response']
    return output
"""
#########################################################################################

#########################################################################################


#########################################################################################
#COMMENT ME TO RUN COMPLETELY "OFFLINE" (STILL NEED TO USE WEB SEARCH API)
#########################################################################################

def get_llm_response(prompt, force_json=False, model = SMALL_MODEL):
    """
    Get a completion from OpenAI's API using GPT-4
    Args:
        prompt (str): The input prompt
        force_json (bool): Whether to force JSON output format
    """
    # Initialize the client with your API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    model_type = None
    if model == MODEL or model == BIG_MODEL:
        model_type = "gpt-4o"
    else:

        model_type = "gpt-4o-mini"
    try:
        # Base parameters
        params = {
            "model": model_type,  # or "gpt-3.5-turbo" for a faster, cheaper option
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 150
        }
        
        # Add response format if JSON is required
        if force_json:
            params["response_format"] = {"type": "json_object"}
            # Add instruction to respond in JSON if not already in prompt
            if not "json" in prompt.lower():
                params["messages"][0]["content"] = f"{prompt} Respond in JSON format."
        
        # Make the API call
        response = client.chat.completions.create(**params)
        
        # Extract the response text
        result = response.choices[0].message.content
        
        # If JSON was requested, validate it
        if force_json:
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return "Failed to parse JSON response"
                
        return result
    
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_llm_json_response(prompt, model=MODEL):
    return get_llm_json_response(prompt=prompt,force_json=True, model=model)

#########################################################################################
#########################################################################################



def determine_informative_local(chunk, claim, use_small=True):
    if use_small:
        m = SMALL_MODEL
    else:
        m = MODEL
    json_res = get_llm_json_response(f'Determine if the following statement is useful in supporting the claim "{claim}". Return a JSON in the form {{"response" : "true"}} or {{"response" : "false"}}. Statement: {chunk}', model=m)
    
    try: 
        # print(f"INFORMATIVE: {dict(json.loads(json_res))}")
        return dict(json.loads(json_res))
    except Exception:
        return {"error": "Error in JSON output"}
    


import dotenv # type: ignore

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
# client.py
import os
from bespokelabs import BespokeLabs

bl = BespokeLabs(
    # This is the default and can be omitted
    auth_token=os.getenv("BESPOKE_API_KEY"),
)
def determine_informative(chunk, claim, use_small=True):
    response = bl.minicheck.factcheck.create(
        claim=claim,
        context=chunk,
    )
    if response.support_prob > 0.6:
        return {"response" : "true"}
    else:
        return {"response" : "false"}


    
def reword_query(claim):
    query = get_llm_response(
        f""" 
        Assume the inputted claim is true, and rephrase into a web search query for evidence that supports the claim. Be careful to use the same tense (past/present/future) as the original claim.:
        Return only the single, brief query for use in searching the web.
        Claim: {claim}    
        Query: """)
    return query

def reverse_claim(claim_to_reverse):
   query = get_llm_response(
       f"""You are a claim reversal expert. Your task is to generate opposite claims while preserving structure and style.

       Rules:
       - Maintain exact sentence structure
       - Keep same verb tenses and grammatical patterns 
       - Flip core meaning by inverting key words 
       - Preserve length and formality level
       - Return ONLY the reversed claim with no additional text

       Examples:
       Original: "Most dogs are friendly."
       Reversed: "Most dogs are hostile."

       Original: "Technology has improved education."  
       Reversed: "Technology has harmed education."

       Original: "Climate change will devastate coastal cities."
       Reversed: "Climate change will benefit coastal cities."

       Claim to reverse: {claim_to_reverse}
       
       Reversed claim: """,    model = BIG_MODEL)
   return query.strip()
def combine_claims(claim, chunk1, chunk2):
    query = get_llm_response(
        f""" Given a claim and related statements, synthesize the statements' main ideas into a single-sentence summary.
            Input:
            Claim: {claim}
            Statements:
            Statement 1: {chunk1}
            Statement 2: {chunk2}
            ...

            Instructions:
            Analyze the claim and statements.
            Synthesize into one clear sentence.
            ONLY RETURN a one-sentence summary encompassing the main points from the claim and statements.""", model=BIG_MODEL)
    return query


def restate_claim(claim, chunk):
    query = get_llm_response(
        f""" Given a claim and a related statement, synthesize the statements main ideas into a single-sentence summary.
            Input:
            Claim: {claim}
            Statements:
            Statement 1: {chunk}
            Instructions:
            Analyze the claim and statements.
            Synthesize into one clear sentence.
            ONLY RETURN a one-sentence summary encompassing the main points from the claim and statements.""", model=BIG_MODEL
                             )
    return query

def get_factoids(claim):
    query = get_llm_response(prompt = 
    f"""For the given claim, generate 5-7 key conceptual phrases that 
    would help verify its accuracy. Focus on broad, categorical details that could be 
    used for semantic matching. Each phrase should be 3-8 words long and capture distinct aspects of the claim. Avoid overly specific details.
    Example:
    Claim: 'Kamala Harris is strict on immigration'
    Output:
    1. Kamala Harris immigration policy voting record
    2. Harris statements on border security
    3. Vice President border enforcement initiatives
    4. Immigration legislation supported by Harris
    5. Harris position on illegal immigration penalties
    Claim:{claim}
    """)
    return query

def restate_evidence(claim, evidence):
    prompt = f"""Claim: {claim}
        Evidence: {evidence}

        Restate the evidence in support of the claim, returning only this restatement as a single sentence:"""

    restatement = get_llm_response(prompt, model=BIG_MODEL)
    return restatement
################################################################################################################
#? EXAMPLE for extract_info_json()
################################################################################################################

#     input_llm = f"""
#     You extract the title from the first part of this chunk, ignoring the author names afterward### Template:
#     {json.dumps(json.loads(schema), indent=4)}
#     ### Example:
#     {{"title": "Amazing new discovery"}}
#     ### Text:
#     {text}
# """
################################################################################################################

def extract_info_json(prompt_with_schema):
    response = ollama.generate(model='nuextract', prompt= prompt_with_schema)['response']
    output = response[response.find("<|end-output|>")+len("<|end-output|>"):]
    print("output")
    print("========================")
    print(output)
    print("========================")

    try:
        return json.loads(output)
    # except json.JSONDecodeError:
    #     print(output)
    #     try:
    #         response = ollama.chat(model='dolphin-llama3', messages=[ #llama3
    #         {
    #         'role': 'user',
    #         'content': prompt_with_schema}])

    #         output = response['response']
            # output = output.replace("<|end-output|>","")

    #         return json.loads(output)
    except json.JSONDecodeError:
        return {"error": "Error in JSON output"}







def get_final_judgement(arg1, arg2, use_small_model=False):
    prompt = f"""Compare the following two arguments and output only a single number (1 or 2) indicating which argument is stronger. Evaluate based on these criteria in order of importance:

    1. Logical validity
    * Sound reasoning structure
    * Absence of logical fallacies
    * Clear cause-and-effect relationships

    2. Resistance to counterarguments
    * Addresses potential objections
    * Accounts for alternative viewpoints
    * Strength of rebuttals

    3. Reasonability of claim
    * Does this seem likely to be true given your assumptions about the world?

    Rules:
    * Output only the number of the stronger argument (1 or 2) in JSON form  ({{"argument": "2"}})
    * If both arguments are exactly equal in strength, output 0
    * Do not include any explanation or justification
    * Do not add any additional text

    Example input:
    Argument 1: [First argument text]
    Argument 2: [Second argument text]
    Example output:
    {{"argument": "1"}}

    Arguments
    Argument: {arg1}
    Argument: {arg2}
    """
    model = "dolphin-llama3:70b"

    if use_small_model:
        model = MODEL

    response = get_llm_response(prompt=prompt, force_json=True, model=BIG_MODEL)
    return response


