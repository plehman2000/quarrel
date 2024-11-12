import gradio as gr
from prover import embed_chunks, filter_chunks_using_vsim, generate_cluster_dict, get_clusters, get_final_args, get_n_informative_chunks, get_webdata_chunks, reverse_claim,reword_query
import pickle

from llm_funcs import get_final_judgement






# import asyncio
# import nest_asyncio
# loop = asyncio.ProactorEventLoop()
# asyncio.set_event_loop(loop)
# get_webdata_chunks("is u s?", 2)



    



def run_prover(proposition_claim, opposition_claim, use_small_model, n_websites, n_chunks_needed_per_cluster, progress=gr.Progress()):
    # await asyncio.wait_for(async_download_websites(prompt), timeout=6)

    try:
        # Input validation
        if not proposition_claim or proposition_claim.strip() == "":
            yield "Error: Proposition claim cannot be empty. Please enter a claim to analyze."
            return
            
        # Initialize status
        status = "Starting"
        proposition_claim=proposition_claim.strip()
        opposition_claim=opposition_claim.strip() if opposition_claim and opposition_claim.strip() else None
        use_small_model=use_small_model
        n_websites=int(n_websites)
        n_chunks_needed_per_cluster=int(n_chunks_needed_per_cluster)
        n_argument_clusters=3
        
        output_text = ""
##################################################################
        url_dict = pickle.load(open("./documents/url_dict.pkl", "rb"))
        max_sampled_chunks_per_cluster = 500

        master_dict = {
            "proposition_claim": proposition_claim,
            "status": "Starting",
            "progress": 0
        }
        progress(master_dict['progress']/100)

        # Generate opposition claim and queries
        if opposition_claim == None:
            opposition_claim = reverse_claim(proposition_claim)
        master_dict.update({
            "opposition_claim": opposition_claim,
            "status": "Generated opposition claim",
            "progress": 10
        })
        progress(master_dict['progress']/100)
        
        proposition_query = reword_query(proposition_claim)
        opposition_query = reword_query(opposition_claim)
        master_dict.update({
            "proposition_query": proposition_query,
            "opposition_query": opposition_query,
            "status": "Generated search queries",
            "progress": 20
        })
        
        progress(master_dict['progress']/100)

        # Get chunks

        prop_chunks_pairs = get_webdata_chunks(proposition_query, n_websites)
        print(prop_chunks_pairs)

        prop_chunks = [x['chunk'] for x in  prop_chunks_pairs]
        master_dict.update({
            "status": "Retrieved proposition web data",
            "progress": 25
        })
        progress(master_dict['progress']/100)

        opp_chunks_pairs = get_webdata_chunks(opposition_query, n_websites)

        opp_chunks = [x['chunk'] for x in  opp_chunks_pairs]

        master_dict.update({
            "status": "Retrieved opposition web data",
            "progress": 30
        })
        progress(master_dict['progress']/100)

        # Embed chunks
        prop_all_chunk_vector_pairs = embed_chunks(prop_chunks)
        master_dict.update({
            "status": "Embedded proposition chunks",
            "progress": 35
        })
        progress(master_dict['progress']/100)

        opp_all_chunk_vector_pairs = embed_chunks(opp_chunks)
        master_dict.update({
            "status": "Embedded opposition chunks",
            "progress": 40
        })
        progress(master_dict['progress']/100)



        prop_reduced_chunk_vector_pairs = filter_chunks_using_vsim(opposition_query, prop_all_chunk_vector_pairs,0.65)
        opp_reduced_chunk_vector_pairs = filter_chunks_using_vsim(opposition_query, opp_all_chunk_vector_pairs, 0.65)
       
       


        prop_sampled_clusters, prop_cluster_ids = get_clusters(prop_reduced_chunk_vector_pairs, n_argument_clusters)
        prop_cluster_dict = generate_cluster_dict(prop_sampled_clusters, prop_reduced_chunk_vector_pairs, prop_cluster_ids)
        master_dict.update({
            "prop_cluster_dict":prop_cluster_dict, #for debuggiign
            "status": "Generated proposition clusters",
            "progress": 45
        })
        progress(master_dict['progress']/100)

        opp_sampled_clusters, opp_cluster_ids = get_clusters(opp_reduced_chunk_vector_pairs, n_argument_clusters)
        opp_cluster_dict = generate_cluster_dict(opp_sampled_clusters, opp_reduced_chunk_vector_pairs, opp_cluster_ids)
        
        
        master_dict.update({
            "opp_cluster_dict":opp_cluster_dict, #for debuggiign
            "status": "Generated opposition clusters",
            "progress": 50
        })
        progress(master_dict['progress']/100)

        


        
        prop_informative_chunks =  get_n_informative_chunks(proposition_claim, prop_cluster_dict,max_sampled_chunks_per_cluster, n_chunks_needed_per_cluster)

        prop_final_args = get_final_args(proposition_claim, prop_cluster_dict, max_sampled_chunks_per_cluster, prop_informative_chunks)
        master_dict.update({
            "prop_final_args": prop_final_args,
            "prop_chunk":prop_informative_chunks,
            "status": "Generated proposition arguments",
            "progress": 60
        })
        progress(master_dict['progress']/100)

        opp_informative_chunks =  get_n_informative_chunks(opposition_claim, opp_cluster_dict,max_sampled_chunks_per_cluster, n_chunks_needed_per_cluster)

        opp_final_args = get_final_args(opposition_claim, opp_cluster_dict, max_sampled_chunks_per_cluster, opp_informative_chunks)
        master_dict.update({
            "opp_final_args": opp_final_args,
            "opp_chunks":opp_informative_chunks,
            "status": "Generated opposition arguments",
            "progress": 70
        })
        progress(master_dict['progress']/100)

        # Format arguments
        arg1_w_claims = f"Claim:{proposition_claim}\n"
        for i, zx in enumerate(prop_final_args):
            arg1_w_claims += f"Premise {i+1}: {zx}\n"

        arg2_w_claims = f"Claim: {opposition_claim}\n"
        for i, zx in enumerate(opp_final_args):
            arg2_w_claims += f"Premise {i+1}: {zx}\n"

        master_dict.update({
            "arg1_w_claims": arg1_w_claims,
            "arg2_w_claims": arg2_w_claims,
            "status": "Formatted arguments",
            "progress": 75
        })
        progress(master_dict['progress']/100)

        # Get final judgment
        final_judge = get_final_judgement(arg1_w_claims, arg2_w_claims, use_small_model=use_small_model)
        idx = int(final_judge['argument'])-1
        choice = [master_dict['proposition_claim'],master_dict['opposition_claim']][idx]
        master_dict['victor'] = choice

        master_dict.update({
            "final_judge": final_judge,
            "status": "Complete",
            "progress": 100,
            "victor" : choice
        })


        output_text = f"Winner: { master_dict['victor']}\n\n" + arg1_w_claims + arg2_w_claims


##################################################################
        yield output_text
    except Exception as e:
        yield f"An error occurred: {str(e)}\nPlease check your inputs and try again."



# Create the Gradio interface with improved styling and validation
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🔍 Argument Prover
    Enter your claims below to analyze and evaluate arguments.
    """)
    
    with gr.Row():
        with gr.Column():
            proposition_input = gr.Textbox(
                label="Proposition Claim",
                placeholder="Enter the main claim (e.g., 'Electric vehicles are better for the environment')",
                lines=3,
                scale=2
            )
            opposition_input = gr.Textbox(
                label="Opposition Claim (optional)",
                placeholder="Enter the opposing claim (optional)",
                lines=3,
                scale=2
            )
            
        with gr.Column():
            with gr.Group():
                gr.Markdown("### Analysis Settings")
                model_checkbox = gr.Checkbox(
                    label="Use Large Model (Recommended)",
                    value=True,
                    info="Large model provides more detailed analysis"
                )
                websites_slider = gr.Slider(
                    minimum=10,
                    maximum=100,
                    # value=50,#TODO Revert to
                    value=10,# DEBUGGING
                    step=10,
                    label="Number of Websites to Search",
                    info="More websites = broader analysis but slower processing"
                )
                chunks_slider = gr.Slider(
                    minimum=2,
                    maximum=20,
                    # value=6,#TODO Revert to
                    value=2,
                    step=2,
                    label="Evidence Chunks per Argument",
                    info="More chunks = stronger evidence but slower processing"
                )
    
    with gr.Row():
        clear_button = gr.Button("Clear", variant="secondary")
        start_button = gr.Button("Start Analysis", variant="primary")
    

    output_text = gr.Textbox(
        label="Analysis Progress and Results",
        lines=20,
        show_copy_button=True,
        placeholder="Results will appear here..."
    )
    
    # Event handlers
    def clear_inputs():
        return {
            proposition_input: "",
            opposition_input: "",
            output_text: ""
        }
    
    start_button.click(
        fn=run_prover,
        inputs=[
            proposition_input,
            opposition_input,
            model_checkbox,
            websites_slider,
            chunks_slider
        ],
        outputs=[output_text]
    )
    
    clear_button.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[proposition_input, opposition_input, output_text]
    )

# Launch the app
if __name__ == "__main__":
    app.queue().launch(show_error=True, share=False)