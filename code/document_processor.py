"""
Document processing for GraphRAG system.
"""
import os
import re
import unicodedata
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: Docling not available. PDF processing will be limited.")

from .config import Config

@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    text_chunk: str
    document_name: str
    document_category: str
    source_file: str
    metadata: Dict[str, Any] = None

class DocumentProcessor:
    """Handles document processing, PDF conversion, and text chunking for any domain."""
    
    def __init__(self, config: Config):
        """Initialize document processor with configuration."""
        self.config = config
        self.converter = self._create_converter() if DOCLING_AVAILABLE else None
        self.splitter = self._create_text_splitter()
    
    def _create_converter(self) -> Optional[DocumentConverter]:
        """Create document converter for PDF processing."""
        if not DOCLING_AVAILABLE:
            return None
            
        # Use configuration to determine accelerator device
        if self.config.document_processing.use_cuda and self.config.document_processing.use_gpu:
            accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
        else:
            accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
            
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create text splitter for chunking."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            separators=self.config.chunking.separators
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove decorative symbols
        text = re.sub(r'[\.\-_]{2,}', ' ', text)
        text = re.sub(r'\.\s', '', text)
        text = re.sub(r'\|', '', text)
        text = re.sub(r'\*{2,}', ' ', text)
        text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
        text = re.sub(r'-\s+', '', text)  # Fix hyphenated breaks
        text = text.replace('\u2022', '-')  # Normalize bullets
        
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        text = ''.join(ch for ch in text if ch.isprintable() and not unicodedata.category(ch).startswith('C'))
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_document_name_from_filename(self, filename: str) -> str:
        """
        Extract document name from filename.
        
        Args:
            filename: PDF filename
            
        Returns:
            Cleaned document name
        """
        filename = filename.replace('.pdf', '')  # Remove extension
        cleaned = re.sub(r'[^a-zA-Z0-9_\-=]', '', filename)  # Keep only allowed chars
        return cleaned
    
    def process_pdf_file(self, file_path: str, document_name: str = None, 
                        document_category: str = None, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Process a single PDF file and return text chunks.
        
        Args:
            file_path: Path to PDF file
            document_name: Name of the document (optional, will use filename if not provided)
            document_category: Category of the document (optional)
            metadata: Additional metadata for the document (optional)
            
        Returns:
            List of text chunks
        """
        if not self.converter:
            raise RuntimeError("PDF converter not available. Install docling for PDF processing.")
        
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Use provided document name or extract from filename
        if document_name is None:
            document_name = self.extract_document_name_from_filename(filename)
        
        # Use provided category or default to filename
        if document_category is None:
            document_category = filename
        
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
        
        doc = self.converter.convert(file_path).document
        raw_text = doc.export_to_text()
        
        print(f'Processing {document_name} (Category: {document_category})')
        print('--------------------------------------')
        
        cleaned = self.clean_text(raw_text)
        raw_chunks = self.splitter.split_text(cleaned)
        
        chunks = []
        for i, chunk in enumerate(raw_chunks):
            # Create a generic formatted chunk
            formatted_chunk = f'Document: {document_name}, Category: {document_category} Text Chunk to follow: {chunk}'
            text_chunk = TextChunk(
                chunk_id=f"{document_name}_c{i}",
                text_chunk=formatted_chunk,
                document_name=document_name,
                document_category=document_category,
                source_file=filename,
                metadata=metadata
            )
            chunks.append(text_chunk)
        
        return chunks
    
    def process_pdf_folder(self, metadata_mapping_file: str = None) -> List[TextChunk]:
        """
        Process all PDF files in the configured folder.
        
        Args:
            metadata_mapping_file: Path to CSV file with document metadata mappings (optional)
            
        Returns:
            List of all text chunks
        """
        all_chunks = []
        pdf_data_path = self.config.document_processing.pdf_data_path
        
        if not os.path.exists(pdf_data_path):
            raise FileNotFoundError(f"PDF data path not found: {pdf_data_path}")
        
        # Load metadata mapping if provided
        metadata_mapping = None
        if metadata_mapping_file or self.config.document_processing.metadata_mapping_file:
            mapping_file = metadata_mapping_file or self.config.document_processing.metadata_mapping_file
            if os.path.exists(mapping_file):
                metadata_mapping = pd.read_csv(mapping_file)
                print(f"Loaded metadata mapping from {mapping_file}")
        
        # Process files based on directory structure
        if os.path.isdir(pdf_data_path):
            # Check if it's a flat directory or organized by categories
            items = os.listdir(pdf_data_path)
            
            if all(os.path.isdir(os.path.join(pdf_data_path, item)) for item in items):
                # Organized by categories (subdirectories)
                all_chunks = self._process_categorized_folder(pdf_data_path, metadata_mapping)
            else:
                # Flat directory structure
                all_chunks = self._process_flat_folder(pdf_data_path, metadata_mapping)
        
        return all_chunks
    
    def _process_categorized_folder(self, pdf_data_path: str, metadata_mapping: pd.DataFrame = None) -> List[TextChunk]:
        """Process PDF files organized in category subdirectories."""
        all_chunks = []
        categories = [f for f in os.listdir(pdf_data_path) if os.path.isdir(os.path.join(pdf_data_path, f))]
        
        for category in categories:
            category_path = os.path.join(pdf_data_path, category)
            pdf_files = [f for f in os.listdir(category_path) if f.lower().endswith('.pdf')]
            
            for pdf_file in pdf_files:
                file_path = os.path.join(category_path, pdf_file)
                document_name = self.extract_document_name_from_filename(pdf_file)
                
                # Get metadata from mapping if available
                metadata = self._get_metadata_from_mapping(document_name, metadata_mapping)
                
                try:
                    chunks = self.process_pdf_file(
                        file_path, 
                        document_name=document_name,
                        document_category=category,
                        metadata=metadata
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {pdf_file}: {e}")
                    continue
        
        return all_chunks
    
    def _process_flat_folder(self, pdf_data_path: str, metadata_mapping: pd.DataFrame = None) -> List[TextChunk]:
        """Process PDF files in a flat directory structure."""
        all_chunks = []
        pdf_files = [f for f in os.listdir(pdf_data_path) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            file_path = os.path.join(pdf_data_path, pdf_file)
            document_name = self.extract_document_name_from_filename(pdf_file)
            
            # Get metadata from mapping if available
            metadata = self._get_metadata_from_mapping(document_name, metadata_mapping)
            
            try:
                chunks = self.process_pdf_file(
                    file_path, 
                    document_name=document_name,
                    metadata=metadata
                )
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue
        
        return all_chunks
    
    def _get_metadata_from_mapping(self, document_name: str, metadata_mapping: pd.DataFrame = None) -> Dict[str, Any]:
        """Extract metadata for a document from the mapping file."""
        if metadata_mapping is None:
            return {}
        
        # Try to find the document in the mapping
        name_field = self.config.document_processing.document_name_field
        category_field = self.config.document_processing.document_category_field
        
        # Look for exact match or partial match
        matches = metadata_mapping[metadata_mapping[name_field].str.contains(document_name, case=False, na=False)]
        
        if not matches.empty:
            row = matches.iloc[0]
            metadata = row.to_dict()
            return metadata
        
        return {}
    
    def save_chunks_to_csv(self, chunks: List[TextChunk], output_file: Optional[str] = None) -> str:
        """
        Save text chunks to CSV file.
        
        Args:
            chunks: List of text chunks
            output_file: Output file path (optional)
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            output_file = self.config.get_output_filename(
                self.config.chunking.chunk_size,
                self.config.chunking.chunk_overlap
            )
        
        # Convert chunks to DataFrame
        data = []
        for chunk in chunks:
            chunk_data = {
                'chunk_id': chunk.chunk_id,
                'text_chunk': chunk.text_chunk,
                'document_name': chunk.document_name,
                'document_category': chunk.document_category,
                'source_file': chunk.source_file
            }
            
            # Add metadata fields if available
            if chunk.metadata:
                for key, value in chunk.metadata.items():
                    chunk_data[f'metadata_{key}'] = value
            
            data.append(chunk_data)
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Exported {len(df)} chunks to {output_file}")
        
        return output_file
    
    def load_chunks_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load text chunks from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with chunks
        """
        df = pd.read_csv(file_path)
        
        # Create a combined identifier if both fields exist
        if 'document_category' in df.columns and 'document_name' in df.columns:
            df['category_document_name'] = df['document_category'] + ' ' + df['document_name']
        
        return df 