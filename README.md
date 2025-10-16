# Embedding pipeline

# Architecture
embedding-pipeline/
├── main.py                   # Pipeline 
├── read_input.py             # Text parsing 
├── save_output.py            # Saving embeddings
├── model_cache/              # Local model storage (auto-created)
├── requirements.txt          # Project dependencies
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
