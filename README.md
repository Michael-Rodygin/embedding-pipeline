# Embedding pipeline

# Architecture
embedding-pipeline/<br>
├── main.py                     <br>
├── read_input.py               <br> 
├── save_output.py              <br>
├── model_cache/                <br>
├── requirements.txt            <br>
└── README.md                 

# Guide

# Clone the repository
git clone https://github.com/Michael-Rodygin/embedding-pipeline.git <br>
cd embedding-pipeline

# Create venv (or whatever environment you want to use)<br>
python -m venv venv_name<br>
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Change PipelineConfig
Locate PipelineConfig class in main.py:<br>
Change:<br>
- model (default='ai-forever/FRIDA')<br>
- model cache location (default='./model_cache')<br>
- processing_batch_size <br>

# Run the pipeline
python main.py

# Prepare input data 
Create a input.txt with input texts 

# Launch main.py
embeddings will be stored in output_embeddings.txt 
