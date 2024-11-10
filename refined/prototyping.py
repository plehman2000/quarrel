import marimo

__generated_with = "0.9.9"
app = marimo.App(width="medium")


@app.cell
def __():
    # from llm_funcs import reword_query, reverse_claim, restate_evidence,restate_claim, get_factoids
    # from text_utils import chunk_text, is_non_informative
    # from web_funcs import download_webpage_html
    # import dotenv
    # import os
    # import uuid
    # from pymojeek import Search
    # from tqdm import tqdm
    # import os
    # import ollama
    # import pandas as pd
    # from sklearn.cluster import k_means
    # import numpy as np
    # from collections import Counter
    # from random import sample
    # from llm_funcs import determine_informative, combine_claims, get_final_judgement
    # from web_funcs import extract_text_from_html_file
    # import asyncio
    # import json
    # from prover import Prover
    return


@app.cell
def __(Prover):
    prover = Prover(
    proposition_claim="Donald Trump is racist",
        opposition_claim = "Donald Trump is not racist",
        use_small_model=False, 
        n_websites=50,
        n_chunks_needed_per_cluster=6)
    return (prover,)


@app.cell
def __(prover):
    out = None
    import time
    start_time = time.time()
    for x in prover.run():
        out = x
        print(out['status'])
        print(f"Time Take: {time.time() - start_time}" )
        start_time = time.time()
    arg1_w_claims = out['arg1_w_claims']
    arg2_w_claims = out['arg2_w_claims']
    print(arg1_w_claims, arg2_w_claims)
    print(f"Winning Claim: {out['victor']}")
    return arg1_w_claims, arg2_w_claims, out, start_time, time, x


@app.cell
def __():
    from prover import get_webdata_chunks, reverse_claim, reword_query
    return get_webdata_chunks, reverse_claim, reword_query


@app.cell
def __():
    import pickle
    return (pickle,)


@app.cell
def __(pickle, reverse_claim, reword_query):
    proposition_claim="Donald Trump is racist"
    opposition_claim="Donald Trump is not racist"
    n_argument_clusters = 6
    n_chunks_needed_per_cluster = 10
    use_small_model = False
    n_websites = 10


    url_dict = pickle.load(open("./documents/url_dict.pkl", "rb"))
    max_sampled_chunks_per_cluster = 500

    master_dict = {
        "proposition_claim": proposition_claim,
        "status": "Starting",
        "progress": 0
    }

    # Generate opposition claim and queries
    if opposition_claim == None:
        opposition_claim = reverse_claim(proposition_claim)
    print(opposition_claim)
    master_dict.update({
        "opposition_claim": opposition_claim,
        "status": "Generated opposition claim",
        "progress": 10
    })

    proposition_query = reword_query(proposition_claim)
    opposition_query = reword_query(opposition_claim)
    master_dict.update({
        "proposition_query": proposition_query,
        "opposition_query": opposition_query,
        "status": "Generated search queries",
        "progress": 20
    })
    return (
        master_dict,
        max_sampled_chunks_per_cluster,
        n_argument_clusters,
        n_chunks_needed_per_cluster,
        n_websites,
        opposition_claim,
        opposition_query,
        proposition_claim,
        proposition_query,
        url_dict,
        use_small_model,
    )


@app.cell
def __(
    get_webdata_chunks,
    master_dict,
    n_websites,
    opposition_query,
    proposition_query,
):
    prop_chunks_pairs = get_webdata_chunks(proposition_query, n_websites)
    prop_chunks = [x['chunk'] for x in  prop_chunks_pairs]
    master_dict.update({
        "status": "Retrieved proposition web data",
        "progress": 30
    })

    opp_chunks_pairs = get_webdata_chunks(opposition_query,n_websites)
    opp_chunks = [x['chunk'] for x in  opp_chunks_pairs]

    master_dict.update({
        "status": "Retrieved opposition web data",
        "progress": 40
    })
    return opp_chunks, opp_chunks_pairs, prop_chunks, prop_chunks_pairs


app._unparsable_cell(
    r"""


        # Embed chunks
        prop_all_chunk_vector_pairs = embed_chunks(prop_chunks)
        master_dict.update({
            \"status\": \"Embedded proposition chunks\",
            \"progress\": 50
        })
        yield master_dict

        opp_all_chunk_vector_pairs = embed_chunks(opp_chunks)
        master_dict.update({
            \"status\": \"Embedded opposition chunks\",
            \"progress\": 60
        })
        yield master_dict



        prop_reduced_chunk_vector_pairs = filter_chunks_using_vsim(opposition_query, prop_all_chunk_vector_pairs,0.65)
        opp_reduced_chunk_vector_pairs = filter_chunks_using_vsim(opposition_query, opp_all_chunk_vector_pairs, 0.65)
       
       


        prop_sampled_clusters, prop_cluster_ids = get_clusters(prop_reduced_chunk_vector_pairs, n_argument_clusters)
        prop_cluster_dict = generate_cluster_dict(prop_sampled_clusters, prop_reduced_chunk_vector_pairs, prop_cluster_ids)
        master_dict.update({
            \"prop_cluster_dict\":prop_cluster_dict, #for debuggiign
            \"status\": \"Generated proposition clusters\",
            \"progress\": 70
        })
        yield master_dict

        opp_sampled_clusters, opp_cluster_ids = get_clusters(opp_reduced_chunk_vector_pairs, n_argument_clusters)
        opp_cluster_dict = generate_cluster_dict(opp_sampled_clusters, opp_reduced_chunk_vector_pairs, opp_cluster_ids)
        
        
        master_dict.update({
            \"opp_cluster_dict\":opp_cluster_dict, #for debuggiign
            \"status\": \"Generated opposition clusters\",
            \"progress\": 80
        })
        yield master_dict

        


        
        prop_informative_chunks =  get_n_informative_chunks(proposition_claim, prop_cluster_dict,max_sampled_chunks_per_cluster, n_chunks_needed_per_cluster)

        prop_final_args = get_final_args(proposition_claim, prop_cluster_dict, max_sampled_chunks_per_cluster, prop_informative_chunks)
        master_dict.update({
            \"prop_final_args\": prop_final_args,
            \"prop_chunk\":prop_informative_chunks,
            \"status\": \"Generated proposition arguments\",
            \"progress\": 85
        })
        yield master_dict

        opp_informative_chunks =  get_n_informative_chunks(opposition_claim, opp_cluster_dict,max_sampled_chunks_per_cluster, n_chunks_needed_per_cluster)

        opp_final_args = get_final_args(opposition_claim, opp_cluster_dict, max_sampled_chunks_per_cluster, opp_informative_chunks)
        master_dict.update({
            \"opp_final_args\": opp_final_args,
            \"opp_chunks\":opp_informative_chunks,
            \"status\": \"Generated opposition arguments\",
            \"progress\": 90
        })
        yield master_dict

        # Format arguments
        arg1_w_claims = f\"Claim:{proposition_claim}\n\"
        for i, zx in enumerate(prop_final_args):
            arg1_w_claims += f\"Premise {i+1}: {zx}\n\"

        arg2_w_claims = f\"Claim: {opposition_claim}\n\"
        for i, zx in enumerate(opp_final_args):
            arg2_w_claims += f\"Premise {i+1}: {zx}\n\"

        master_dict.update({
            \"arg1_w_claims\": arg1_w_claims,
            \"arg2_w_claims\": arg2_w_claims,
            \"status\": \"Formatted arguments\",
            \"progress\": 95
        })
        yield master_dict

        # Get final judgment
        final_judge = get_final_judgement(arg1_w_claims, arg2_w_claims, use_small_model=use_small_model)
        idx = int(final_judge['argument'])-1
        choice = [master_dict['proposition_claim'],master_dict['opposition_claim']][idx]
        master_dict['victor'] = choice

        master_dict.update({
            \"final_judge\": final_judge,
            \"status\": \"Complete\",
            \"progress\": 100,
            \"victor\" : choice
        })


    """,
    name="__"
)


if __name__ == "__main__":
    app.run()
