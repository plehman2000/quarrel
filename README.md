#  :mag: Quarrel 
![alt text](https://github.com/plehman2000/quarrel/blob/main/assets/im1.png)

An AI-powered tool that analyzes opposing claims by gathering and evaluating evidence from web sources to determine the stronger argument.

![alt text](https://github.com/plehman2000/quarrel/blob/main/assets/im2.png)

## Features

- Automated web scraping and evidence gathering
- Semantic clustering of evidence using embeddings
- Intelligent argument construction and evaluation 
- Support for both proposition and opposition claims
- Configurable analysis parameters

## Usage

1. Enter your proposition claim
2. Optionally enter an opposition claim (auto-generated if not provided)
3. Configure analysis settings:
   - Model size (Large/Small)
   - Number of websites to search (10-100)
   - Evidence chunks per argument (2-20)
4. Click "Start Analysis" to begin evaluation

## Technical Details

- Uses DuckDuckGo search API for web scraping, html_text for text extraction
- Employs Nomic embeddings for text chunk filtering with adaptive vector similarity threshold
- Clustered evidence using k-means
- Evaluates evidence relevance using Bespoke Minicheck
- Reduces and combines evidence into coherent arguments using Dolphin-Llama-3, a censorship-resistant llama variant (in offline mode)
- Makes final judgment based on argument strength

## Requirements

```python
# Core dependencies
ollama (if running offline)
numpy
scikit-learn
simsimd
duckduckgo_search
python-dotenv
tqdm
```

## Environment Variables

Required environment variables:
```
OPENAI_API_KEY
BESPOKE_API_KEY
URL_DICTIONARY_FILEPATH 
WEBCHUNK_DICTIONARY_FILEPATH  
WEBFILES_FILEPATH
```

## Development

To run locally:
 
```bash
# Clone repository
git clone https://github.com/yourusername/argument-prover.git

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your values

# Run the application
python app.py
```
NOTE: By default, running this will use the Bespoke and OpenAI API, respectively. However, it can be run entirely locally if you have decent GPU. Simply edit the marked lines of the "get_llm_response" functions in llm_funcs.py. 

