"""
Main GraphRAG class that orchestrates the entire workflow.
"""
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .config import Config
from .azure_client import AzureClient
from .document_processor import DocumentProcessor, TextChunk
from .graph_processor import GraphProcessor, Entity, Relationship
from .embedding_processor import EmbeddingProcessor
from .prompts import PromptTemplates

class GraphRAG:
    """
    Main GraphRAG class that orchestrates the entire workflow for processing
    credit card documents and extracting entities and relationships.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize GraphRAG system.
        
        Args:
            config: Configuration object (optional, will create default if not provided)
        """
        self.config = config or Config()
        self.azure_client = AzureClient(self.config)
        self.document_processor = DocumentProcessor(self.config)
        self.graph_processor = GraphProcessor(self.config, self.azure_client)
        self.embedding_processor = EmbeddingProcessor(self.config, self.azure_client)
        
        # Ensure data directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        Path(self.config.output_data).mkdir(exist_ok=True)
        Path(self.config.graph_data_folder).mkdir(exist_ok=True)
    
    def process_documents(self, card_mapping_file: str) -> str:
        """
        Process PDF documents and create text chunks.
        
        Args:
            card_mapping_file: Path to CSV file with card name mappings
            
        Returns:
            Path to saved chunks CSV file
        """
        print("Starting document processing...")
        chunks = self.document_processor.process_pdf_folder(card_mapping_file)
        output_file = self.document_processor.save_chunks_to_csv(chunks)
        print(f"Document processing completed. Output saved to: {output_file}")
        return output_file
    
    def load_chunks(self, chunks_file: str) -> pd.DataFrame:
        """
        Load text chunks from CSV file.
        
        Args:
            chunks_file: Path to chunks CSV file
            
        Returns:
            DataFrame with chunks
        """
        return self.document_processor.load_chunks_from_csv(chunks_file)
    
    def extract_graph_data(self, chunks_file: str, version: str = "v4") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract entities and relationships from text chunks.
        
        Args:
            chunks_file: Path to chunks CSV file
            version: Version suffix for output files
            
        Returns:
            Tuple of (entities_df, relationships_df)
        """
        print("Starting graph data extraction...")
        
        # Load chunks
        data = self.load_chunks(chunks_file)
        text_chunks = list(data['text_chunk'])
        chunk_index = list(data['bank_card_name'])
        
        # Extract entities and relationships
        entities, relationships = self.graph_processor.extract_entities_and_relationships(
            text_chunks, chunk_index
        )
        
        # Apply mappings and clean data
        entities = self.graph_processor.apply_entity_mappings(entities)
        relationships = self.graph_processor.apply_relationship_mappings(relationships)
        
        # Create DataFrames
        entities_df, relationships_df = self.graph_processor.create_dataframes(entities, relationships)
        
        # Save data
        self.graph_processor.save_graph_data(entities_df, relationships_df, version)
        
        print(f"Graph data extraction completed. Found {len(entities)} entities and {len(relationships)} relationships.")
        return entities_df, relationships_df
    
    def generate_embeddings(self, relationships_file: str, version: str = "v4") -> str:
        """
        Generate embeddings for relationship descriptions.
        
        Args:
            relationships_file: Path to relationships CSV file
            version: Version suffix for output files
            
        Returns:
            Path to saved embeddings file
        """
        print("Starting embedding generation...")
        
        # Load relationships
        relationships_df = pd.read_csv(relationships_file)
        
        # Generate embeddings
        embeddings = self.embedding_processor.generate_relationship_embeddings(relationships_df)
        
        # Save embeddings
        output_file = self.embedding_processor.save_embeddings(embeddings, version)
        
        print(f"Embedding generation completed. Output saved to: {output_file}")
        return output_file
    
    def analyze_similarities(self, 
                           relationships_file: str, 
                           embeddings_file: str,
                           thresholds: Optional[List[float]] = None) -> Dict[float, Dict[str, pd.DataFrame]]:
        """
        Analyze similarities between relationships.
        
        Args:
            relationships_file: Path to relationships CSV file
            embeddings_file: Path to embeddings pickle file
            thresholds: List of similarity thresholds
            
        Returns:
            Dictionary mapping thresholds to card-based similarity pairs
        """
        print("Starting similarity analysis...")
        
        # Load data
        data = pd.read_csv(relationships_file)
        embeddings = self.embedding_processor.load_embeddings()
        
        # Add embeddings to DataFrame
        data_with_embeddings = self.embedding_processor.add_embeddings_to_dataframe(data, embeddings)
        
        # Find similarity pairs
        similarity_pairs = self.embedding_processor.find_similarity_pairs_multiple_thresholds(
            data_with_embeddings, thresholds
        )
        
        # Print statistics
        statistics = self.embedding_processor.get_similarity_statistics(data_with_embeddings, thresholds)
        for threshold, count in statistics.items():
            print(f"Threshold {threshold}: {count} similarity pairs found")
        
        print("Similarity analysis completed.")
        return similarity_pairs
    
    def run_complete_pipeline(self, 
                             card_mapping_file: str, 
                             chunks_file: Optional[str] = None,
                             version: str = "v4") -> Dict[str, Any]:
        """
        Run the complete GraphRAG pipeline.
        
        Args:
            card_mapping_file: Path to CSV file with card name mappings
            chunks_file: Path to existing chunks file (optional, will process PDFs if not provided)
            version: Version suffix for output files
            
        Returns:
            Dictionary containing all results
        """
        print("Starting complete GraphRAG pipeline...")
        results = {}
        
        # Step 1: Process documents (if chunks file not provided)
        if chunks_file is None:
            print("\n=== Step 1: Document Processing ===")
            chunks_file = self.process_documents(card_mapping_file)
        else:
            print(f"\n=== Step 1: Using existing chunks file: {chunks_file} ===")
        
        results['chunks_file'] = chunks_file
        
        # Step 2: Extract graph data
        print("\n=== Step 2: Graph Data Extraction ===")
        entities_df, relationships_df = self.extract_graph_data(chunks_file, version)
        results['entities_df'] = entities_df
        results['relationships_df'] = relationships_df
        
        # Step 3: Generate embeddings
        print("\n=== Step 3: Embedding Generation ===")
        relationships_file = f'{self.config.graph_data_folder}relationships_{version}.csv'
        embeddings_file = self.generate_embeddings(relationships_file, version)
        results['embeddings_file'] = embeddings_file
        
        # Step 4: Analyze similarities
        print("\n=== Step 4: Similarity Analysis ===")
        similarity_pairs = self.analyze_similarities(relationships_file, embeddings_file)
        results['similarity_pairs'] = similarity_pairs
        
        print("\n=== GraphRAG Pipeline Completed Successfully ===")
        return results
    
    def compare_relationship_similarity(self, text_1: str, text_2: str) -> bool:
        """
        Compare two relationship descriptions for similarity using LLM.
        
        Args:
            text_1: First relationship description
            text_2: Second relationship description
            
        Returns:
            True if similar, False otherwise
        """
        prompt = PromptTemplates.format_similarity_comparison_prompt(text_1, text_2)
        response = self.azure_client.get_chat_response(prompt)
        
        if response and response.choices:
            result = response.choices[0].message.content.strip().lower()
            return result == "true"
        
        return False
    
    def get_entity_statistics(self, entities_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about extracted entities.
        
        Args:
            entities_df: Entities DataFrame
            
        Returns:
            Dictionary with entity statistics
        """
        stats = {
            'total_entities': len(entities_df),
            'unique_entities': entities_df['entity_name'].nunique(),
            'entity_types': entities_df['entity_type'].value_counts().to_dict(),
            'cards_covered': entities_df['card_name'].nunique(),
            'entities_per_card': entities_df.groupby('card_name')['entity_name'].count().to_dict()
        }
        return stats
    
    def get_relationship_statistics(self, relationships_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about extracted relationships.
        
        Args:
            relationships_df: Relationships DataFrame
            
        Returns:
            Dictionary with relationship statistics
        """
        stats = {
            'total_relationships': len(relationships_df),
            'unique_source_entities': relationships_df['source_entity'].nunique(),
            'unique_target_entities': relationships_df['target_entity'].nunique(),
            'cards_covered': relationships_df['card_name'].nunique(),
            'relationships_per_card': relationships_df.groupby('card_name').size().to_dict(),
            'strength_distribution': relationships_df['relationship_strength'].value_counts().sort_index().to_dict()
        }
        return stats
    
    def export_results(self, 
                      entities_df: pd.DataFrame, 
                      relationships_df: pd.DataFrame,
                      similarity_pairs: Dict[float, Dict[str, pd.DataFrame]],
                      output_dir: str = "results") -> None:
        """
        Export all results to files.
        
        Args:
            entities_df: Entities DataFrame
            relationships_df: Relationships DataFrame
            similarity_pairs: Similarity analysis results
            output_dir: Output directory
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Export DataFrames
        entities_df.to_csv(f"{output_dir}/entities.csv", index=False)
        relationships_df.to_csv(f"{output_dir}/relationships.csv", index=False)
        
        # Export statistics
        entity_stats = self.get_entity_statistics(entities_df)
        relationship_stats = self.get_relationship_statistics(relationships_df)
        
        import json
        with open(f"{output_dir}/entity_statistics.json", 'w') as f:
            json.dump(entity_stats, f, indent=2)
        
        with open(f"{output_dir}/relationship_statistics.json", 'w') as f:
            json.dump(relationship_stats, f, indent=2)
        
        # Export similarity pairs
        for threshold, card_pairs in similarity_pairs.items():
            threshold_dir = f"{output_dir}/similarity_pairs_{threshold}"
            Path(threshold_dir).mkdir(exist_ok=True)
            
            for card_name, pairs_df in card_pairs.items():
                if len(pairs_df) > 0:
                    safe_card_name = card_name.replace('/', '_').replace('\\', '_')
                    pairs_df.to_csv(f"{threshold_dir}/{safe_card_name}.csv", index=False)
        
        print(f"Results exported to {output_dir}/") 