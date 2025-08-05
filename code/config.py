"""
Configuration management for GraphRAG system.
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv

@dataclass
class AzureConfig:
    """Azure OpenAI configuration settings. Default model is gpt-4o-mini and default embedding model is text-embedding-3-small. 
    If you want to use a different model and API version, you can set the environment variables accordingly.
    """
    endpoint: str = None
    model_name: str = "gpt-4o-mini"
    deployment: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_deployment: str = "text-embedding-3-small"
    api_version: str = "2024-12-01-preview"
    embedding_api_version: str = "2024-02-01"
    system_prompt: str = "You are an expert graph assistant that extracts entities and relationships from document text chunks accurately."
    
    def __post_init__(self):
        # Load environment variables
        load_dotenv()
        
        # Set endpoint from environment variable if not provided
        if self.endpoint is None:
            self.endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', self.endpoint)


@dataclass
class ChunkingConfig:
    """Text chunking configuration. Default chunk size and overlap are 4500 and 500 respectively. Change this based on your use case."""
    chunk_size: int = 4500
    chunk_overlap: int = 500
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ".", " ", ""]


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    max_retries: int = 3
    retry_delay: int = 30
    similarity_threshold: float = 0.7
    similarity_thresholds: List[float] = None
    
    def __post_init__(self):
        if self.similarity_thresholds is None:
            self.similarity_thresholds = [0.85, 0.90, 0.95]


@dataclass
class DocumentProcessingConfig:
    """Document processing configuration."""
    use_cuda: bool = True
    use_gpu: bool = True
    pdf_data_path: str = "../data/"
    metadata_mapping_file: str = None
    document_category_field: str = "category"
    document_name_field: str = "name"
    
    def __post_init__(self):
        # Load environment variables
        load_dotenv()
        
        # Allow environment variable override
        cuda_env = os.getenv('USE_CUDA', 'true').lower()
        gpu_env = os.getenv('USE_GPU', 'true').lower()
        pdf_path_env = os.getenv('PDF_DATA_PATH', self.pdf_data_path)
        mapping_file_env = os.getenv('METADATA_MAPPING_FILE', self.metadata_mapping_file)
        
        if cuda_env in ['false', '0', 'no']:
            self.use_cuda = False
        if gpu_env in ['false', '0', 'no']:
            self.use_gpu = False
        if pdf_path_env:
            self.pdf_data_path = pdf_path_env
        if mapping_file_env:
            self.metadata_mapping_file = mapping_file_env


@dataclass
class EntityTypes:
    """Entity type definitions. Modify this based on your use case."""
    types: List[str] = None
    
    def __post_init__(self):
        if self.types is None:
            # Generic entity types that work for most use cases
            self.types = [
                "Document", "Category", "Topic", "Concept", "Person", 
                "Organization", "Location", "Date", "Amount", "Percentage",
                "Policy", "Rule", "Benefit", "Requirement", "Process",
                "Contact", "Terms", "Others"
            ]


class Config:
    """Main configuration class for GraphRAG system."""
    
    def __init__(self):
        load_dotenv()
        
        self.azure = AzureConfig()
        self.chunking = ChunkingConfig()
        self.processing = ProcessingConfig()
        self.document_processing = DocumentProcessingConfig()
        self.entity_types = EntityTypes()
        
        # API Key
        self.openai_api_key = os.getenv('OpenAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI_API_KEY environment variable not found")
        
        # File paths - use document processing config for PDF data path
        self.data = self.document_processing.pdf_data_path
        self.output_data = "../output_data/"
        self.graph_data_folder = "../output_data/graph_data/"
        
        # Entity name mappings for cleaning and normalization
        self.entity_name_mappings = {}
        self.entity_name_corrections = {}
        self.document_name_mappings = {}
    
    def get_output_filename(self, chunk_size: int, chunk_overlap: int) -> str:
        """Generate output filename for chunks."""
        return f"{self.output_data}chunks_{chunk_size}_{chunk_overlap}.csv"
    
    def get_graph_output_filename(self, version: str = "v1") -> str:
        """Generate output filename for graph data."""
        return f"{self.output_data}graph_index_output_{version}.pkl"
    
    def get_relationships_embeddings_filename(self, version: str = "v1") -> str:
        """Generate output filename for relationship embeddings."""
        return f"{self.output_data}relationships_embeddings_{version}.pkl" 