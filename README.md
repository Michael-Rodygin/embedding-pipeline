# Embedding pipeline

# Architecture
embedding-pipeline/<br>
├── main.py                   # Pipeline<br>
├── read_input.py             # Text parsing<br> 
├── save_output.py            # Saving embeddings<br>
├── model_cache/              # Local model storage (auto-created)<br>
├── requirements.txt          # Project dependencies<br>
└── README.md                 # Project documentation

# Guide

# Clone the repository
git clone https://github.com/Michael-Rodygin/embedding-pipeline.git
cd embedding-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py

# Prepare input data 
Create a input.txt with input texts 

# Launch main.py
embeddings will be stored in output_embeddings.txt 
