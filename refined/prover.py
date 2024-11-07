from llm_funcs import reword_query, reverse_claim, restate_evidence,restate_claim
from text_utils import chunk_text, is_non_informative
from web_funcs import download_webpage_html
import dotenv
import os
import uuid
from pymojeek import Search
from tqdm import tqdm
import os
import ollama
import pickle
from sklearn.cluster import k_means
import numpy as np
from collections import Counter
from random import sample
from llm_funcs import determine_informative, combine_claims, get_final_judgement
from web_funcs import extract_text_from_html_file
import asyncio
import json
DEBUG = False
# def extract_text_from_html_file_ML(file_path):
#     html_content = None
#     normalized_path = os.path.normpath(file_path)
#     try:
#         with open(normalized_path, 'r', encoding='utf-8') as file:
#             html_content = file.read()
#     except FileNotFoundError:
#         print(f"HTML file not found at: {normalized_path}")
#     except IOError as e:
#         print(f"Error reading HTML file: {str(e)}")

#     if html_content:
#         return convert_html_markdown(html_content)
#     else:
#         return None



dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
# Load environment variables from .env file
api_key = os.getenv("PYMOJEEK_API_KEY")
mojk_client = Search(api_key=api_key)

from duckduckgo_search import DDGS

def get_claim_sources(query, use_mojk=False, n_websites=10):
    results_final = []
    if use_mojk:
        results = mojk_client.search(query, count=min(10, n_websites)) #10 is the limit for current plan
        for x in results:
            results_final.append({'url':x.url})
    else:
        #use DDGS 
        results = DDGS().text(query, max_results=n_websites)
        for x in results:
            results_final.append({'url':x['href']})


    if DEBUG:
        print(f"Found {len(results_final)} results for query '{query}'")
    return results_final

def save_source_pages(results):
    claimid = str(uuid.uuid4())
    filedir = f"./documents/claim-{claimid}/"
    os.makedirs(filedir)
    
    #URL Dictionary logic
    url_dict_filepath = './documents/url_dict.pkl'
    site_id_dict = {}
    if os.path.exists(url_dict_filepath):
        with open(url_dict_filepath, 'rb') as file:
            site_id_dict = pickle.load(file)

    urls = []
    filenames = []
    for x in results:
        site_id = str(uuid.uuid4())
        site_id_dict[site_id] = str(x['url'])
        urls.append(x['url']) 
        filenames.append(site_id)
    try:
        asyncio.run(download_webpage_html(urls, filenames, save_folder=filedir))
    except Exception:
        print("Exception thrown during download_webpage_html")
    # with open('./documents/url_dict.pkl', 'wb') as file:
    #     pickle.dump(site_id_dict, file)
    with open(url_dict_filepath, 'wb') as file:
        pickle.dump(site_id_dict, file)

    return filedir
def ingest_source_pages(filedir):
    all_chunks = []
    num_files = 0
    for file in tqdm(os.listdir(filedir),desc="Ingesting source pages..."):
        text = extract_text_from_html_file(filedir + file)
        num_files = num_files +1
        chunks = chunk_text(text)
        for i,c in enumerate(chunks):
            # if not is_non_informative(c):
            all_chunks.append({"chunk":c, "source":filedir, "chunk_index": i})
    return all_chunks

def get_webdata_chunks(query, n_websites):
    results = get_claim_sources(query, use_mojk=False, n_websites=n_websites)
    filedir = save_source_pages(results)
    chunks = ingest_source_pages(filedir)
    return chunks


def embed_chunks(sourced_chunks, is_query=False):
    all_chunk_vector_pairs = []
    for sourced_chunk in tqdm(sourced_chunks, desc="Embedding chunks..."):
        if len(sourced_chunk) > 15:
            # print(sourced_chunk)
            if is_query:
                sourced_chunk = "search_query: " + sourced_chunk
            else:
                sourced_chunk = "search_document: " + sourced_chunk
            embedding = ollama.embeddings(model="nomic-embed-text", prompt=  sourced_chunk)["embedding"]
            all_chunk_vector_pairs.append([sourced_chunk, embedding])
    return all_chunk_vector_pairs



def get_clusters(chunk_vector_pairs, n_argument_clusters):
    print(chunk_vector_pairs[0])
    clustered = k_means(np.array([x[1] for x in chunk_vector_pairs]), n_argument_clusters)
    _, cluster_ids, _ = clustered
    # Take N biggest clusters
    sampled_clusters = [x[0] for x in sorted(Counter(clustered[1]).items(), key = lambda x : x[1], reverse=True)]
    return sampled_clusters, cluster_ids

def generate_cluster_dict(sampled_clusters, chunk_vector_pairs, cluster_ids):
    cluster_to_chunk = {}
    # Initialize empty lists for all sampled clusters
    for cluster in sampled_clusters:
        cluster_to_chunk[cluster] = []
    
    # Assign chunks to their respective clusters
    for (chunk, _), cluster_id in zip(chunk_vector_pairs, cluster_ids):
        if cluster_id in sampled_clusters:  # Only add chunks for sampled clusters
            cluster_to_chunk[cluster_id].append(chunk)
            
    return cluster_to_chunk



def get_n_informative_chunks(claim,cluster_to_chunk_dict,max_sampled_chunks_per_cluster, n_chunks_needed_per_cluster):
    informative_chunks = {}
    for clust_i in tqdm(cluster_to_chunk_dict.keys(), desc=f"Getting {n_chunks_needed_per_cluster} informative chunks per cluster"):
        if max_sampled_chunks_per_cluster < len(cluster_to_chunk_dict[clust_i]):
            sampled_chunks = sample(list(cluster_to_chunk_dict[clust_i]), max_sampled_chunks_per_cluster)
        else:
            sampled_chunks = list(cluster_to_chunk_dict[clust_i])
        informative_chunks[clust_i] = []
        ct = 0
        for chu in sampled_chunks:
            informative = determine_informative(chu, claim)
            if 'response' in informative:
                if informative['response'].lower() == 'true':
                    # print(f"Info chunk: {chu}")
                    informative_chunks[clust_i].append(chu)
                    ct +=1
                    if DEBUG:
                        print(f"{ct} chunk(s) found")

            if ct >=n_chunks_needed_per_cluster:
                print(f"Enough Info Chunk(s) Found! ({len(informative_chunks[clust_i])})")
                break
    return informative_chunks

def reduce_chunks(claim, chunks):
    if len(chunks) == 0:
        return None
    print(f"Reducing {len(chunks)} chunks...")
    print(chunks)
    intermediate_summaries = chunks
    while len(intermediate_summaries) != 1:
        temp = []
        for zx in range(0,len(intermediate_summaries), 2):
            if zx+1 < len(intermediate_summaries):
                combined_claim = combine_claims(claim, intermediate_summaries[zx], intermediate_summaries[zx+1])
                temp.append(combined_claim)
            else:
                temp.append(restate_evidence(claim,intermediate_summaries[zx]))
        intermediate_summaries = temp
        # print(intermediate_summaries)
        # print(len(intermediate_summaries))

    final_argument = intermediate_summaries[0]
    return final_argument


def get_final_args(claim,cluster_to_chunk_dict,max_sampled_chunks_per_cluster, informative_chunks):

    final_args = []
    for cl in informative_chunks:
        final_chunk = reduce_chunks(claim, informative_chunks[cl])
        if final_chunk:
            final_arg = restate_evidence(claim,final_chunk)
            final_args.append(final_arg)
            print("Chunks Reduced!")
    return final_args




import simsimd

def filter_chunks_using_vsim(query, all_chunk_vector_pairs, thresh=0.65):
    filter_embeddings = embed_chunks([query], is_query=True)
    filter_embedding = np.array(filter_embeddings[0][1])
    # get vector similarities for elimination
    similarities = [] 
    for i, chunky in tqdm(enumerate(all_chunk_vector_pairs), total=len(all_chunk_vector_pairs)):
        vector = np.array(chunky[1])
        sim = 1- simsimd.cosine(vector, filter_embedding)
        similarities.append(sim)
    similarities = np.array(similarities)
    indeces = np.where(similarities > 0.65)[0] 
    
    return [all_chunk_vector_pairs[idx] for idx in indeces]




import pickle
class Prover():
    def __init__(self, proposition_claim, opposition_claim=None,n_argument_clusters = 3,n_chunks_needed_per_cluster = 10, use_small_model=True, n_websites=20):
        self.proposition_claim = proposition_claim
        self.opposition_claim= opposition_claim
        self.n_argument_clusters = n_argument_clusters
        self.n_chunks_needed_per_cluster = n_chunks_needed_per_cluster
        self.use_small_model=use_small_model
        self.n_websites = n_websites
        assert n_chunks_needed_per_cluster%2 == 0

        self.url_dict = pickle.load(open("./documents/url_dict.pkl", "rb"))

    def run(self):
        

        


        max_sampled_chunks_per_cluster = 500

        master_dict = {
            "proposition_claim": self.proposition_claim,
            "status": "Starting",
            "progress": 0
        }
        yield master_dict

        # Generate opposition claim and queries
        if self.opposition_claim == None:
            self.opposition_claim = reverse_claim(self.proposition_claim)
        print(self.opposition_claim)
        master_dict.update({
            "opposition_claim": self.opposition_claim,
            "status": "Generated opposition claim",
            "progress": 10
        })
        yield master_dict
        
        proposition_query = reword_query(self.proposition_claim)
        opposition_query = reword_query(self.opposition_claim)
        master_dict.update({
            "proposition_query": proposition_query,
            "opposition_query": opposition_query,
            "status": "Generated search queries",
            "progress": 20
        })
        yield master_dict

        # Get chunks
        prop_chunks_pairs = get_webdata_chunks(proposition_query, self.n_websites)
        prop_chunks = [x['chunk'] for x in  prop_chunks_pairs]
        master_dict.update({
            "status": "Retrieved proposition web data",
            "progress": 30
        })
        yield master_dict

        opp_chunks_pairs = get_webdata_chunks(opposition_query,self.n_websites)
        opp_chunks = [x['chunk'] for x in  opp_chunks_pairs]

        master_dict.update({
            "status": "Retrieved opposition web data",
            "progress": 40
        })
        yield master_dict

        # Embed chunks
        prop_all_chunk_vector_pairs = embed_chunks(prop_chunks)
        master_dict.update({
            "status": "Embedded proposition chunks",
            "progress": 50
        })
        yield master_dict

        opp_all_chunk_vector_pairs = embed_chunks(opp_chunks)
        master_dict.update({
            "status": "Embedded opposition chunks",
            "progress": 60
        })
        yield master_dict



        prop_reduced_chunk_vector_pairs = filter_chunks_using_vsim(opposition_query, prop_all_chunk_vector_pairs,0.65)
        opp_reduced_chunk_vector_pairs = filter_chunks_using_vsim(opposition_query, opp_all_chunk_vector_pairs, 0.65)
       
       


        prop_sampled_clusters, prop_cluster_ids = get_clusters(prop_reduced_chunk_vector_pairs, self.n_argument_clusters)
        prop_cluster_dict = generate_cluster_dict(prop_sampled_clusters, prop_reduced_chunk_vector_pairs, prop_cluster_ids)
        master_dict.update({
            "prop_cluster_dict":prop_cluster_dict, #for debuggiign
            "status": "Generated proposition clusters",
            "progress": 70
        })
        yield master_dict

        opp_sampled_clusters, opp_cluster_ids = get_clusters(opp_reduced_chunk_vector_pairs, self.n_argument_clusters)
        opp_cluster_dict = generate_cluster_dict(opp_sampled_clusters, opp_reduced_chunk_vector_pairs, opp_cluster_ids)
        
        
        master_dict.update({
            "opp_cluster_dict":opp_cluster_dict, #for debuggiign
            "status": "Generated opposition clusters",
            "progress": 80
        })
        yield master_dict

        


        
        prop_informative_chunks =  get_n_informative_chunks(self.proposition_claim, prop_cluster_dict,max_sampled_chunks_per_cluster, self.n_chunks_needed_per_cluster)

        prop_final_args = get_final_args(self.proposition_claim, prop_cluster_dict, max_sampled_chunks_per_cluster, prop_informative_chunks)
        master_dict.update({
            "prop_final_args": prop_final_args,
            "prop_chunk":prop_informative_chunks,
            "status": "Generated proposition arguments",
            "progress": 85
        })
        yield master_dict

        opp_informative_chunks =  get_n_informative_chunks(self.opposition_claim, opp_cluster_dict,max_sampled_chunks_per_cluster, self.n_chunks_needed_per_cluster)

        opp_final_args = get_final_args(self.opposition_claim, opp_cluster_dict, max_sampled_chunks_per_cluster, opp_informative_chunks)
        master_dict.update({
            "opp_final_args": opp_final_args,
            "opp_chunks":opp_informative_chunks,
            "status": "Generated opposition arguments",
            "progress": 90
        })
        yield master_dict

        # Format arguments
        arg1_w_claims = f"Claim:{self.proposition_claim}\n"
        for i, zx in enumerate(prop_final_args):
            arg1_w_claims += f"Premise {i+1}: {zx}\n"

        arg2_w_claims = f"Claim: {self.opposition_claim}\n"
        for i, zx in enumerate(opp_final_args):
            arg2_w_claims += f"Premise {i+1}: {zx}\n"

        master_dict.update({
            "arg1_w_claims": arg1_w_claims,
            "arg2_w_claims": arg2_w_claims,
            "status": "Formatted arguments",
            "progress": 95
        })
        yield master_dict

        # Get final judgment
        final_judge = get_final_judgement(arg1_w_claims, arg2_w_claims, use_small_model=self.use_small_model)
        idx = int(final_judge['argument'])-1
        choice = [master_dict['proposition_claim'],master_dict['opposition_claim']][idx]
        master_dict['victor'] = choice

        master_dict.update({
            "final_judge": final_judge,
            "status": "Complete",
            "progress": 100,
            "victor" : choice
        })



        yield master_dict




