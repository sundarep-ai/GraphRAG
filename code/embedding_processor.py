"""
Embedding processing for GraphRAG system.
"""
import time
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available. Using NumPy for similarity calculations.")

from .config import Config
from .azure_client import AzureClient

@dataclass
class SimilarityPair:
    """Represents a pair of similar relationships."""
    index_1: int
    index_2: int
    similarity: float
    text_1: str
    source_entity_1: str
    target_entity_1: str
    text_2: str
    source_entity_2: str
    target_entity_2: str

class EmbeddingProcessor:
    """Handles embedding generation and similarity analysis."""
    
    def __init__(self, config: Config, azure_client: AzureClient):
        """Initialize embedding processor with configuration and Azure client."""
        self.config = config
        self.azure_client = azure_client
    
    def generate_relationship_embeddings(self, relationships_df: pd.DataFrame) -> List[List[float]]:
        """
        Generate embeddings for relationship descriptions.
        
        Args:
            relationships_df: DataFrame containing relationships
            
        Returns:
            List of embedding vectors
        """
        total = len(relationships_df)
        output = []
        start_time = time.time()
        
        print(f"Embedding text chunks............")
        
        for num, text_chunk in enumerate(relationships_df['relationship_description']):
            embedding = self.azure_client.get_embedding(text_chunk)
            output.append(embedding)
            
            elapsed_time = int(time.time() - start_time)
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            print(f"Chunks Embedded: {num+1}/{total}\nElapsed time: {elapsed_str}")
        
        return output
    
    def save_embeddings(self, embeddings: List[List[float]], version: str = "v4") -> str:
        """
        Save embeddings to pickle file.
        
        Args:
            embeddings: List of embedding vectors
            version: Version suffix for filename
            
        Returns:
            Path to saved file
        """
        output_file = self.config.get_relationships_embeddings_filename(version)
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)
        return output_file
    
    def load_embeddings(self, version: str = "v4") -> List[List[float]]:
        """
        Load embeddings from pickle file.
        
        Args:
            version: Version suffix for filename
            
        Returns:
            List of embedding vectors
        """
        input_file = self.config.get_relationships_embeddings_filename(version)
        with open(input_file, 'rb') as f:
            return pickle.load(f)
    
    def get_high_similarity_pairs(self, 
                                 df: pd.DataFrame, 
                                 threshold: float = 0.95, 
                                 vector_column: str = 'embedding_vector',
                                 text_column: str = 'relationship_description',
                                 source: str = 'source_entity',
                                 target: str = 'target_entity') -> pd.DataFrame:
        """
        Find pairs of relationships with high similarity scores.
        
        Args:
            df: DataFrame containing relationships and embeddings
            threshold: Similarity threshold (0.0 to 1.0)
            vector_column: Column name containing embedding vectors
            text_column: Column name containing relationship descriptions
            source: Column name for source entity
            target: Column name for target entity
            
        Returns:
            DataFrame with similarity pairs
        """
        vectors = np.array(df[vector_column].tolist())
        
        if CUPY_AVAILABLE:
            vectors = cp.array(vectors)
            vectors = vectors / cp.linalg.norm(vectors, axis=1, keepdims=True)
            sim_matrix = cp.dot(vectors, vectors.T)
            sim_matrix = cp.clip(sim_matrix, -1.0, 1.0)
            
            i_idx, j_idx = list(df.index), list(df.index)
            i_indices, j_indices = cp.triu_indices(sim_matrix.shape[0], k=1)
            sim_values = sim_matrix[i_indices, j_indices]
            
            mask = sim_values > threshold
            i_filtered = i_indices[mask].get()
            j_filtered = j_indices[mask].get()
            scores = sim_values[mask].get()
        else:
            # Use NumPy if CuPy is not available
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            sim_matrix = np.dot(vectors, vectors.T)
            sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
            
            i_idx, j_idx = list(df.index), list(df.index)
            i_indices, j_indices = np.triu_indices(sim_matrix.shape[0], k=1)
            sim_values = sim_matrix[i_indices, j_indices]
            
            mask = sim_values > threshold
            i_filtered = i_indices[mask]
            j_filtered = j_indices[mask]
            scores = sim_values[mask]
        
        index_dict_i = {i: i_idx[i] for i in range(len(i_idx))}
        index_dict_j = {j: j_idx[j] for j in range(len(j_idx))}
        
        matches = pd.DataFrame({
            "index_1": [index_dict_i[i] for i in i_filtered],
            "index_2": [index_dict_j[j] for j in j_filtered],
            "similarity": scores,
            "text_1": df.iloc[i_filtered][text_column].values,
            'source_entity_1': df.iloc[i_filtered][source].values,
            'target_entity_1': df.iloc[i_filtered][target].values,
            "text_2": df.iloc[j_filtered][text_column].values,
            'source_entity_2': df.iloc[j_filtered][source].values,
            'target_entity_2': df.iloc[j_filtered][target].values,
        })
        
        return matches.sort_values(by="similarity", ascending=False)
    
    def find_similarity_pairs_by_card(self, 
                                     data: pd.DataFrame, 
                                     threshold: float = 0.90) -> Dict[str, pd.DataFrame]:
        """
        Find similarity pairs grouped by card name.
        
        Args:
            data: DataFrame containing relationships and embeddings
            threshold: Similarity threshold
            
        Returns:
            Dictionary mapping card names to similarity pairs
        """
        pairs = {}
        for card, group in data.groupby("card_name"):
            pairs[card] = self.get_high_similarity_pairs(group, threshold=threshold)
        return pairs
    
    def find_similarity_pairs_multiple_thresholds(self, 
                                                 data: pd.DataFrame, 
                                                 thresholds: Optional[List[float]] = None) -> Dict[float, Dict[str, pd.DataFrame]]:
        """
        Find similarity pairs for multiple thresholds.
        
        Args:
            data: DataFrame containing relationships and embeddings
            thresholds: List of similarity thresholds
            
        Returns:
            Dictionary mapping thresholds to card-based similarity pairs
        """
        if thresholds is None:
            thresholds = self.config.processing.similarity_thresholds
        
        all_pairs = {}
        for threshold in thresholds:
            all_pairs[threshold] = self.find_similarity_pairs_by_card(data, threshold)
        
        return all_pairs
    
    def extract_all_pairs(self, 
                         data: pd.DataFrame, 
                         threshold: float = 0.9) -> List[Tuple[str, str]]:
        """
        Extract all similarity pairs as text tuples.
        
        Args:
            data: DataFrame containing relationships and embeddings
            threshold: Similarity threshold
            
        Returns:
            List of text pairs
        """
        pairs = []
        for card, group in data.groupby("card_name"):
            sim_pairs = self.get_high_similarity_pairs(group, threshold=threshold)
            text_1 = sim_pairs.text_1
            text_2 = sim_pairs.text_2
            pairs.extend([(text_1[i], text_2[i]) for i in range(sim_pairs.shape[0])])
        
        return pairs
    
    def get_similarity_statistics(self, 
                                 data: pd.DataFrame, 
                                 thresholds: Optional[List[float]] = None) -> Dict[float, int]:
        """
        Get statistics on similarity pairs for different thresholds.
        
        Args:
            data: DataFrame containing relationships and embeddings
            thresholds: List of similarity thresholds
            
        Returns:
            Dictionary mapping thresholds to total pair counts
        """
        if thresholds is None:
            thresholds = self.config.processing.similarity_thresholds
        
        all_pairs = self.find_similarity_pairs_multiple_thresholds(data, thresholds)
        statistics = {}
        
        for threshold in thresholds:
            total = 0
            for card in data.card_name.unique():
                total += all_pairs[threshold][card].shape[0]
            statistics[threshold] = total
        
        return statistics
    
    def add_embeddings_to_dataframe(self, 
                                   df: pd.DataFrame, 
                                   embeddings: List[List[float]]) -> pd.DataFrame:
        """
        Add embedding vectors to DataFrame.
        
        Args:
            df: DataFrame to add embeddings to
            embeddings: List of embedding vectors
            
        Returns:
            DataFrame with embeddings added
        """
        df_copy = df.copy()
        df_copy['embedding_vector'] = embeddings
        return df_copy 