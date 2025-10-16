from sentence_transformers import SentenceTransformer
from read_input import parse_input
from save_output import save_embeddings
import torch
import logging
import os
from typing import List, Optional
from dataclasses import dataclass
import time

GLOBAL_PARAM_use_gpu = False


@dataclass
class PipelineConfig:
    model_id: str = 'ai-forever/FRIDA'
    local_model_cache: str = './model_cache'
    compute_device: str = 'auto'
    processing_batch_size: int = 32
    enable_progress_bar: bool = True


class ModelCacheManager:
    def __init__(self, cache_dir: str = './model_cache'):
        self.cache_directory = cache_dir
        self._ensure_cache_directory()

    def _ensure_cache_directory(self):
        os.makedirs(self.cache_directory, exist_ok=True)

    def get_model_path(self, model_id: str) -> str:
        safe_model_name = model_id.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_directory, safe_model_name)

    def is_model_cached(self, model_id: str) -> bool:
        model_path = self.get_model_path(model_id)
        return os.path.exists(model_path) and os.path.isdir(model_path)

    def cache_model(self, model: SentenceTransformer, model_id: str):
        try:
            model_path = self.get_model_path(model_id)
            model.save(model_path)
            logging.info(f"Model successfully cached to: {model_path}")
        except Exception as e:
            logging.warning(f"Model caching failed: {str(e)}")


class EmbeddingEngine:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self._model_instance: Optional[SentenceTransformer] = None
        self.cache_manager = ModelCacheManager(self.config.local_model_cache)
        self._initialize_logging()

    def _initialize_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _acquire_compute_device(self) -> str:
        if self.config.compute_device == 'auto':
            if torch.cuda.is_available() and GLOBAL_PARAM_use_gpu:
                device = 'cuda'
                self.logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                self.logger.warning("GPU not available, defaulting to CPU processing")
        else:
            device = self.config.compute_device

        return device

    def _load_model_from_cache(self, model_id: str) -> Optional[SentenceTransformer]:
        if self.cache_manager.is_model_cached(model_id):
            try:
                cached_model_path = self.cache_manager.get_model_path(model_id)
                target_device = self._acquire_compute_device()

                self.logger.info(f"Loading cached model from: {cached_model_path}")
                model = SentenceTransformer(cached_model_path, device=target_device)

                self.logger.info("Model successfully loaded from local cache")
                return model
            except Exception as e:
                self.logger.warning(f"Failed to load from cache: {str(e)}")
        return None

    def _download_and_cache_model(self, model_id: str) -> SentenceTransformer:
        target_device = self._acquire_compute_device()

        self.logger.info(f"Downloading model: {model_id}")
        model = SentenceTransformer(model_id, device=target_device)

        self.cache_manager.cache_model(model, model_id)

        return model

    def _initialize_model_architecture(self) -> SentenceTransformer:
        cached_model = self._load_model_from_cache(self.config.model_id)
        if cached_model:
            return cached_model

        self.logger.info("Model not found in cache, downloading...")
        return self._download_and_cache_model(self.config.model_id)

    def get_model(self) -> SentenceTransformer:
        if self._model_instance is None:
            self.logger.info("Initializing model architecture...")
            self._model_instance = self._initialize_model_architecture()
            self.logger.info("Model architecture successfully deployed")
        return self._model_instance

    def execute_embedding_pipeline(self, input_path: str, output_path: str) -> None:
        pipeline_start = time.perf_counter()

        model = self.get_model()

        self.logger.info("Initiating document ingestion phase")
        document_corpus = parse_input(input_path)

        if not document_corpus:
            self.logger.error("Document corpus empty - terminating pipeline")
            return

        self._log_corpus_statistics(document_corpus)

        self.logger.info("Commencing embedding computation")
        embedding_tensor = model.encode(
            document_corpus,
            batch_size=self.config.processing_batch_size,
            show_progress_bar=self.config.enable_progress_bar,
            normalize_embeddings=True
        )

        self._persist_embeddings(embedding_tensor, output_path)

        pipeline_duration = time.perf_counter() - pipeline_start
        self.logger.info(f"Pipeline completed in {pipeline_duration:.2f} seconds")

    def _log_corpus_statistics(self, documents: List[str]):
        total_chunks = len(documents)
        sample_preview = documents[0][:75] + "..." if documents[0] else "Empty"

        self.logger.info(f"Corpus Statistics: {total_chunks} documents ingested")
        self.logger.info(f"Sample Document: {sample_preview}")
        self.logger.info(f"Document Identifiers: {len([f'doc_{i}' for i in range(total_chunks)])}")

    def _persist_embeddings(self, embeddings, output_path: str):
        try:
            embedding_matrix = embeddings.tolist()

            self.logger.info(f"Embedding Tensor Dimensions: {embeddings.shape}")
            self.logger.info(f"Serializing to storage: {output_path}")

            save_embeddings(
                embedding_matrix,
                output_path,
                separator="$$$"
            )

            self.logger.info("Embedding persistence completed successfully")

        except Exception as serialization_error:
            self.logger.error(f"Embedding serialization failed: {str(serialization_error)}")
            raise


def main():
    pipeline_config = PipelineConfig(
        local_model_cache='./model_cache',
        compute_device='auto',
        processing_batch_size=32,
        enable_progress_bar=True
    )

    embedding_engine = EmbeddingEngine(pipeline_config)
    embedding_engine.execute_embedding_pipeline(
        input_path="input.txt",
        output_path="output_embeddings.txt"
    )


if __name__ == "__main__":
    main()
