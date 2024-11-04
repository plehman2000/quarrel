import marimo

__generated_with = "0.9.10"
app = marimo.App(width="medium")


@app.cell
def __():
    # from prover import Prover
    # prover = Prover()
    # claim = "The minecraft youtuber Dream is a pedophile"
    # oppclaim = "The minecraft youtuber Dream is not a pedophile"
    # out = None
    # import time
    # start_time = time.time()
    # for x in prover.run(proposition_claim=claim,opposition_claim = oppclaim, use_small_model=False):
    #     out = x
    #     print(out['status'])
    #     print(time.time() - start_time )
    #     start_time = time.time()
    # arg1_w_claims = out['arg1_w_claims']
    # arg2_w_claims = out['arg2_w_claims']
    # print(arg1_w_claims, arg2_w_claims)
    # print(f"Winning Claim: {out['victor']}")
    return


@app.cell
def __():
    return


@app.cell
def __():
    from llm_funcs import reword_query, reverse_claim, restate_evidence,restate_claim, get_factoids
    from text_utils import chunk_text, is_non_informative
    from web_funcs import download_webpage_html
    import dotenv
    import os
    import uuid
    from pymojeek import Search
    from tqdm import tqdm
    import os
    import ollama
    from sklearn.cluster import k_means
    import numpy as np
    from collections import Counter
    from random import sample
    from llm_funcs import determine_informative, combine_claims, get_final_judgement, convert_html_markdown
    from web_funcs import extract_text_from_html_file
    import asyncio
    import json
    return (
        Counter,
        Search,
        asyncio,
        chunk_text,
        combine_claims,
        convert_html_markdown,
        determine_informative,
        dotenv,
        download_webpage_html,
        extract_text_from_html_file,
        get_factoids,
        get_final_judgement,
        is_non_informative,
        json,
        k_means,
        np,
        ollama,
        os,
        restate_claim,
        restate_evidence,
        reverse_claim,
        reword_query,
        sample,
        tqdm,
        uuid,
    )


@app.cell
def __():
    import prover
    return (prover,)


@app.cell
def __(reverse_claim):
    import pickle
    url_dict = pickle.load(open("./documents/url_dict.pkl", "rb"))
    proposition_claim = "The minecraft youtuber Dream is a pedophile"
    opposition_claim = "The minecraft youtuber Dream is not a pedophile"
    n_argument_clusters = 3
    n_chunks_needed_per_cluster = 10
    use_small_model=True
    master_dict = {}
    if opposition_claim == None:
         opposition_claim = reverse_claim( proposition_claim)
    return (
        master_dict,
        n_argument_clusters,
        n_chunks_needed_per_cluster,
        opposition_claim,
        pickle,
        proposition_claim,
        url_dict,
        use_small_model,
    )


@app.cell
def __(master_dict, opposition_claim, proposition_claim, reword_query):
    master_dict.update({
        "opposition_claim":  opposition_claim,
        "status": "Generated opposition claim",
        "progress": 10
    })

    proposition_query = reword_query( proposition_claim)
    opposition_query = reword_query( opposition_claim)
    master_dict.update({
        "proposition_query": proposition_query,
        "opposition_query": opposition_query,
        "status": "Generated search queries",
        "progress": 20
    })

    return opposition_query, proposition_query


@app.cell
def __(master_dict):
    print(master_dict)

    return


@app.cell
def __(master_dict, opposition_query, proposition_query, prover):
    # Generate opposition claim and queries


    # Get chunks
    prop_chunks_pairs = prover.get_webdata_chunks(proposition_query)
    prop_chunks = [x['chunk'] for x in  prop_chunks_pairs]
    master_dict.update({
        "status": "Retrieved proposition web data",
        "progress": 30
    })

    opp_chunks_pairs = prover.get_webdata_chunks(opposition_query)
    opp_chunks = [x['chunk'] for x in  opp_chunks_pairs]

    master_dict.update({
        "status": "Retrieved opposition web data",
        "progress": 40
    })

    return opp_chunks, opp_chunks_pairs, prop_chunks, prop_chunks_pairs


@app.cell
def __(mo):
    mo.md(r"""# Prototype for Faster Chunk winnowing""")
    return


@app.cell
def __(prop_chunks):
    prop_chunks[0]
    return


@app.cell
def __(master_dict, opp_chunks, prop_chunks, prover):

    # Embed chunks
    prop_all_chunk_vector_pairs = prover.embed_chunks(prop_chunks)
    master_dict.update({
        "status": "Embedded proposition chunks",
        "progress": 50
    })

    opp_all_chunk_vector_pairs = prover.embed_chunks(opp_chunks)
    master_dict.update({
        "status": "Embedded opposition chunks",
        "progress": 60
    })

    return opp_all_chunk_vector_pairs, prop_all_chunk_vector_pairs


@app.cell
def __():
    return


@app.cell
def __(np, opposition_query, prover):
    # Get embedding of filter chunks

    # q = get_factoids(proposition_claim)
    # print(q)

    # filter_embeddings = prover.embed_chunks([q])
    filter_embeddings = prover.embed_chunks([opposition_query], is_query=True)
    filter_embedding = np.array(filter_embeddings[0][1])

    return filter_embedding, filter_embeddings


@app.cell
def __(filter_embedding):
    import simsimd

    similarity = 1-simsimd.cosine(filter_embedding, filter_embedding)
    print(similarity)
    return similarity, simsimd


@app.cell
def __(filter_embedding, np, opp_all_chunk_vector_pairs, simsimd, tqdm):
    # get vector similarities for elimination
    similarities = [] 
    for i, chunky in tqdm(enumerate(opp_all_chunk_vector_pairs), total=len(opp_all_chunk_vector_pairs)):
        vector = np.array(chunky[1])
        sim = 1- simsimd.cosine(vector, filter_embedding)
        similarities.append(sim)

    similarities = np.array(similarities)
    return chunky, i, sim, similarities, vector


@app.cell
def __(np, opp_all_chunk_vector_pairs, similarities):
    indeces = np.where(similarities > 0.6)[0]  # Returns 3
    print(len(indeces))
    for idx in indeces:
        x =     opp_all_chunk_vector_pairs[idx][0]
        print(x)
        # break
    return idx, indeces, x


@app.cell
def __():
    # opp_all_chunk_vector_pairs[0][1]
    return


app._unparsable_cell(
    r"""

    # embedd chunks
    check for chunks similar to generated abstract facts
    """,
    name="__"
)


@app.cell
def __(mo):
    mo.md(r"""# Rest of Code""")
    return


app._unparsable_cell(
    r"""



            prop_sampled_clusters, prop_cluster_ids = get_clusters(prop_all_chunk_vector_pairs, n_argument_clusters)
            prop_cluster_dict = generate_cluster_dict(prop_sampled_clusters, prop_all_chunk_vector_pairs, prop_cluster_ids)
            master_dict.update({
                \"status\": \"Generated proposition clusters\",
                \"progress\": 70
            })
            yield master_dict

            opp_sampled_clusters, opp_cluster_ids = get_clusters(opp_all_chunk_vector_pairs, n_argument_clusters)
            opp_cluster_dict = generate_cluster_dict(opp_sampled_clusters, opp_all_chunk_vector_pairs, opp_cluster_ids)
            master_dict.update({
                \"status\": \"Generated opposition clusters\",
                \"progress\": 80
            })
            yield master_dict

            
            prop_informative_chunks =  get_n_informative_chunks( proposition_claim, prop_cluster_dict,max_sampled_chunks_per_cluster, n_chunks_needed_per_cluster)

            prop_final_args = get_final_args( proposition_claim, prop_cluster_dict, max_sampled_chunks_per_cluster, prop_informative_chunks)
            master_dict.update({
                \"prop_final_args\": prop_final_args,
                \"prop_chunk\":prop_informative_chunks,
                \"status\": \"Generated proposition arguments\",
                \"progress\": 85
            })
            yield master_dict

            opp_informative_chunks =  get_n_informative_chunks( opposition_claim, opp_cluster_dict,max_sampled_chunks_per_cluster, n_chunks_needed_per_cluster)

            opp_final_args = get_final_args( opposition_claim, opp_cluster_dict, max_sampled_chunks_per_cluster, opp_informative_chunks)
            master_dict.update({
                \"opp_final_args\": opp_final_args,
                \"opp_chunks\":opp_informative_chunks,
                \"status\": \"Generated opposition arguments\",
                \"progress\": 90
            })
            yield master_dict

            # Format arguments
            arg1_w_claims = f\"Claim:{ proposition_claim}\n\"
            for i, zx in enumerate(prop_final_args):
                arg1_w_claims += f\"Premise {i+1}: {zx}\n\"

            arg2_w_claims = f\"Claim: { opposition_claim}\n\"
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



            yield master_dict





    """,
    name="__"
)


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
