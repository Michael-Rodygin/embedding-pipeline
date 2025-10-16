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
```bash
git clone https://github.com/Michael-Rodygin/embedding-pipeline.git
```
```bash
cd embedding-pipeline
```

# Create venv (or whatever environment you want to use)<br>
```bash
python -m venv venv_name
```
```bash
venv_name\Scripts\activate
```

# Install dependencies
```bash
pip install -r requirements.txt
```

# (Optional) Change PipelineConfig
Locate PipelineConfig class in main.py:<br>
Change:<br>
- model (default='ai-forever/FRIDA')<br>
- model cache location (default='./model_cache')<br>
- processing_batch_size <br>

# Prepare input data 
Create an input.txt with input texts

# Run the pipeline
```bash
python main.py
```

# Result
Embeddings will be stored in output_embeddings.txt file
