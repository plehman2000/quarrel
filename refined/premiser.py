

from gc import enable
import dearpygui.dearpygui as dpg

from llm_funcs import get_llm_response, extract_info_json
import json
import time

json_extraction_prompt = f"""
    You extract the premises from the following list of premises. There should only be one 'premises' tag in the returned JSON. Always return a filled JSON. ALWAYS INCLUDE A "premises" tag!. Only return the JSON
    ### Template:
    {{"premises":[""]}}
    ### Example:
    {{"premises":["This guy is not very nice", "this guy is mean"]}}
    ### Text:\n
"""
def get_conclusion_premise(conclusion, n=5):
    prompt = f"""
    Suppose the following conclusion is a proven fact. Give {n} premises that provide an argument that this is true in complete sentences. Only return these premises. Return each premise in a valid JSON, like {{"premise1" : "he is mean", {{"premise2": "..."}}}}. Only return JSON.
    {conclusion}
    """
    premises = get_llm_response(prompt)
    return premises

def get_inversion(conclusion, premises):
    prompt = f"""
    Given the following premises and conclusion:
    {conclusion}\n
    {premises}
    Reframe each premise so it does not support the original conclusion. Only return these new, inverted premises. Return each premises in a valid json, like {{"premise1" : "he is mean", {{"premise2": "..."}}}}. Only return JSON.
    """
    premises = get_llm_response(prompt)
    return premises






def spawn_premise_windows_dpg(premise_list, tag_prefix, x=0,y=0, new_window_width=400, new_window_height=75):
    # Get the position and size of the start_window to arrange new windows
    start_window_pos = dpg.get_item_pos("start_window")
    start_window_width = dpg.get_item_width("start_window")

    # Calculate the position for the new windows
    new_window_x = start_window_pos[0] + start_window_width + x
    new_window_y = start_window_pos[1] + y
    print(premise_list)
    for i, prem in enumerate(premise_list):
        with dpg.window(tag=f"{tag_prefix}_premise_{i}", label=f"Premise {i + 1}", pos=[new_window_x, new_window_y], width=new_window_width, height=new_window_height):
            # Add the premise text inside the window
            dpg.add_text(prem, wrap=0)
        # Adjust vertical position for the next window (stacked below the previous one)
        new_window_y += new_window_height + 10
        time.sleep(0.1)

    
def get_premises_dpg(conclusion):
    
    # Add a loading indicator to show processing
    loading_indicator = dpg.add_loading_indicator(style=1,tag="loader_og", parent="start_window", color=(52, 235, 186), indent=2, speed=3)

    # Get premises and their inversions
    premises = get_conclusion_premise(conclusion)
    inverted_premises = get_inversion(conclusion, premises)
    print("premis\n" + premises)
    premises_list = json.loads(premises).values()
    inverted_premises_list = json.loads(inverted_premises).values()
    print(type(premises_list))
    new_window_width=400
    new_window_height=75
    # Extract structured information as JSON
    # premises_json = extract_info_json(json_extraction_prompt + premises)
    spawn_premise_windows_dpg(premises_list, tag_prefix = "proposition", new_window_width=new_window_width, new_window_height=new_window_height)
    # inverted_premises_json = extract_info_json(json_extraction_prompt + inverted_premises)
    # Reset X position for the inverted premises
    spawn_premise_windows_dpg(inverted_premises_list, tag_prefix="opposition", x=new_window_width, new_window_width=new_window_width, new_window_height=new_window_height)
    # Remove the loading indicator after the premises are created
    dpg.delete_item("loader_og")
    return premises_list, inverted_premises_list

def premise_flow_dpg():
    # Get the conclusion input from a widget (e.g., a text box)
    conclusion = dpg.get_value("prompt_input")
    premises_list, inverted_premises_list = get_premises_dpg(conclusion)
    return premises_list, inverted_premises_list


# from langchain_community.tools import DuckDuckGoSearchResults

# tool = DuckDuckGoSearchResults()
# results = tool.invoke("Obama")
# print(results)



import spacy
import json
def get_entities(text):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    # entities = [(ent.text, ent.label_) for ent in doc.ents 
    entities = [[ent.text, ent.label_] for ent in doc.ents 
                if ent.label_ not in ["MONEY", "TIME", "DATE", "CARDINAL", "PERCENT", "QUANTITY", "ORDINAL"]]

    return entities




def argue_against(conclusion):
    prompt = f"""
    Write the strongest possible 3 sentence argument against the following argument
    {conclusion}
    """
    premises = get_llm_response(prompt)
    return premises


def trim_leading_whitespace(s):
    lines = s.splitlines()
    if lines and lines[0].strip() == '':
        return ''.join(lines[1:])
    return s.lstrip()

import json
def get_premises(conclusion):
    premises = get_conclusion_premise(conclusion)
    inverted_premises = get_inversion(conclusion, premises)
    premises_list = [trim_leading_whitespace(x) for x in list(json.loads(premises).values())]
    inverted_premises_list = [trim_leading_whitespace(x) for x in list(json.loads(inverted_premises).values())]
    return premises_list, inverted_premises_list