import gradio as gr
import asyncio
import nest_asyncio
from prover import Prover

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def run_prover_async(proposition_claim, opposition_claim, use_small_model, n_websites, n_chunks_needed_per_cluster, progress=gr.Progress()):
    try:
        # Input validation
        if not proposition_claim or proposition_claim.strip() == "":
            yield "Error: Proposition claim cannot be empty. Please enter a claim to analyze."
            return
            
        # Initialize status
        status = "Starting"
        
        # Create Prover instance
        prover = Prover(
            proposition_claim=proposition_claim.strip(),
            opposition_claim=opposition_claim.strip() if opposition_claim and opposition_claim.strip() else None,
            use_small_model=use_small_model,
            n_websites=int(n_websites),
            n_chunks_needed_per_cluster=int(n_chunks_needed_per_cluster)
        )
        
        # Run the prover and yield updates
        async for result in prover.run():
            status = result["status"]
            progress_value = result.get("progress", 0)
            
            # Update progress bar
            progress(progress_value/100)
            
            # Prepare output string with clear formatting
            output_text = f"Status: {status}\nProgress: {progress_value}%\n\n"
            
            if "proposition_query" in result:
                output_text += f"Search Queries:\n"
                output_text += f"- Proposition: {result['proposition_query']}\n"
                if result.get('opposition_query'):
                    output_text += f"- Opposition: {result['opposition_query']}\n"
                output_text += "\n"
                
            if "prop_final_args" in result:
                output_text += "📊 Proposition Arguments:\n"
                for i, arg in enumerate(result['prop_final_args'], 1):
                    output_text += f"{i}. {arg}\n"
                output_text += "\n"
                
            if "opp_final_args" in result:
                output_text += "📊 Opposition Arguments:\n"
                for i, arg in enumerate(result['opp_final_args'], 1):
                    output_text += f"{i}. {arg}\n"
                output_text += "\n"
                
            if "victor" in result:
                output_text += f"\n🏆 WINNING ARGUMENT: {result['victor']}\n"
                if "final_judge" in result:
                    output_text += f"\nReasoning:\n{result['final_judge']['explanation']}"
            
            yield output_text
            
    except Exception as e:
        yield f"An error occurred: {str(e)}\nPlease check your inputs and try again."

def run_prover_sync(*args, **kwargs):
    """Wrapper to run the async function in the event loop"""
    async def process_generator():
        async for output in run_prover_async(*args, **kwargs):
            return output  # Return the first result
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(process_generator())

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
                    value=50,
                    step=10,
                    label="Number of Websites to Search",
                    info="More websites = broader analysis but slower processing"
                )
                chunks_slider = gr.Slider(
                    minimum=2,
                    maximum=20,
                    value=6,
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
    
    # Connect the components using async handling
    start_button.click(
        fn=run_prover_sync,
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
    app.queue().launch(show_error=True)