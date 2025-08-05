"""
Graph processing for GraphRAG system.
"""
import re
import time
import pickle
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .config import Config
from .azure_client import AzureClient
from .prompts import PromptTemplates

@dataclass
class Entity:
    """Represents an extracted entity."""
    entity_name: str
    entity_type: str
    entity_description: str
    document_name: str

@dataclass
class Relationship:
    """Represents an extracted relationship."""
    source_entity: str
    target_entity: str
    relationship_description: str
    relationship_strength: int
    document_name: str

class GraphProcessor:
    """Handles graph processing, entity extraction, and relationship analysis for any domain."""
    
    def __init__(self, config: Config, azure_client: AzureClient):
        """Initialize graph processor with configuration and Azure client."""
        self.config = config
        self.azure_client = azure_client
    
    def process_output(self, text: str) -> str:
        """
        Process and clean output text.
        
        Args:
            text: Raw output text
            
        Returns:
            Cleaned text
        """
        return text.replace("{tuple_delimiter}", "|||").replace("{record_delimiter}", "").replace("{completion_delimiter}", "")
    
    def extract_entities_and_relationships(self, text_chunks: List[str], chunk_index: List[str]) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from text chunks.
        
        Args:
            text_chunks: List of text chunks to process
            chunk_index: List of document names corresponding to chunks
            
        Returns:
            Tuple of (entities, relationships)
        """
        output = []
        total = len(text_chunks)
        start_time = time.time()
        
        print(f"Processing text chunks............")
        
        for num, text_chunk in enumerate(text_chunks):
            prompt_content = PromptTemplates.format_entity_extraction_prompt(text_chunk)
            response = self.azure_client.get_chat_response(prompt_content)
            
            if response and response.choices:
                output.append(response.choices[0].message.content)
            else:
                print("No valid response received.")
            
            elapsed_time = int(time.time() - start_time)
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            print(f"Chunks Processed: {num+1}/{total}\nElapsed time: {elapsed_str}")
        
        return self._parse_output(output, chunk_index)
    
    def _parse_output(self, output: List[str], chunk_index: List[str]) -> Tuple[List[Entity], List[Relationship]]:
        """
        Parse the output to extract entities and relationships.
        
        Args:
            output: List of raw outputs
            chunk_index: List of document names
            
        Returns:
            Tuple of (entities, relationships)
        """
        records = []
        entity_tuples = []
        relationship_tuples = []
        
        for row, text in enumerate(output):
            temp = [r.strip() for r in self.process_output(text).strip().split("\n") if r.strip()]
            records.extend([r[:-1] + '|||' + chunk_index[row] + r[-1:] for r in temp])
        
        for r in records:
            if r.startswith('("entity"'):
                content = re.search(r'\("entity"\|\|\|(.*)\)', r)
                if content:
                    fields = content.group(1).split("|||")
                    if len(fields) == 4:
                        entity_tuples.append(fields)
            elif r.startswith('("relationship"'):
                content = re.search(r'\("relationship"\|\|\|(.*)\)', r)
                if content:
                    fields = content.group(1).split("|||")
                    if len(fields) == 5:
                        relationship_tuples.append(fields)
        
        entities = [Entity(*fields) for fields in entity_tuples]
        relationships = [Relationship(*fields) for fields in relationship_tuples]
        
        return entities, relationships
    
    def clean_entity_name(self, name: str) -> str:
        """
        Clean entity name by removing special characters and normalizing.
        
        Args:
            name: Raw entity name
            
        Returns:
            Cleaned entity name
        """
        if pd.isna(name):
            return name
        
        cleaned = re.sub(r'[_\-\\/%+]', ' ', name)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip().upper()
    
    def apply_entity_mappings(self, entities: List[Entity]) -> List[Entity]:
        """
        Apply entity name mappings and corrections.
        
        Args:
            entities: List of entities
            
        Returns:
            Updated entities
        """
        for entity in entities:
            # Apply entity name corrections
            if entity.entity_name in self.config.entity_name_corrections:
                entity.entity_name = self.config.entity_name_corrections[entity.entity_name]
            
            # Apply entity name mappings
            if entity.entity_name in self.config.entity_name_mappings:
                entity.entity_name = self.config.entity_name_mappings[entity.entity_name]
            
            # Apply document name mappings
            if entity.entity_name in self.config.document_name_mappings:
                entity.entity_name = self.config.document_name_mappings[entity.entity_name]
            
            # Clean entity name
            entity.entity_name = self.clean_entity_name(entity.entity_name)
        
        return entities
    
    def apply_relationship_mappings(self, relationships: List[Relationship]) -> List[Relationship]:
        """
        Apply entity name mappings to relationships.
        
        Args:
            relationships: List of relationships
            
        Returns:
            Updated relationships
        """
        for relationship in relationships:
            # Apply entity name mappings
            if relationship.source_entity in self.config.entity_name_mappings:
                relationship.source_entity = self.config.entity_name_mappings[relationship.source_entity]
            if relationship.target_entity in self.config.entity_name_mappings:
                relationship.target_entity = self.config.entity_name_mappings[relationship.target_entity]
            
            # Apply document name mappings
            if relationship.source_entity in self.config.document_name_mappings:
                relationship.source_entity = self.config.document_name_mappings[relationship.source_entity]
            if relationship.target_entity in self.config.document_name_mappings:
                relationship.target_entity = self.config.document_name_mappings[relationship.target_entity]
            
            # Clean entity names
            relationship.source_entity = self.clean_entity_name(relationship.source_entity)
            relationship.target_entity = self.clean_entity_name(relationship.target_entity)
        
        return relationships
    
    def remove_non_ascii(self, text: str) -> str:
        """
        Remove non-ASCII characters from text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        return re.sub(r'[^\x20-\x7E]', '', text)
    
    def create_dataframes(self, entities: List[Entity], relationships: List[Relationship]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create pandas DataFrames from entities and relationships.
        
        Args:
            entities: List of entities
            relationships: List of relationships
            
        Returns:
            Tuple of (entities_df, relationships_df)
        """
        # Create entities DataFrame
        entities_data = [
            {
                'entity_name': e.entity_name,
                'entity_type': e.entity_type,
                'entity_description': e.entity_description,
                'document_name': e.document_name
            }
            for e in entities
        ]
        entities_df = pd.DataFrame(entities_data)
        
        # Create relationships DataFrame
        relationships_data = [
            {
                'source_entity': r.source_entity,
                'target_entity': r.target_entity,
                'relationship_description': r.relationship_description,
                'relationship_strength': r.relationship_strength,
                'document_name': r.document_name
            }
            for r in relationships
        ]
        relationships_df = pd.DataFrame(relationships_data)
        
        # Clean relationship descriptions
        relationships_df['relationship_description'] = relationships_df['relationship_description'].apply(self.remove_non_ascii)
        
        # Add entity types to relationships
        entity_type_map = entities_df[['entity_name', 'entity_type']].drop_duplicates(subset='entity_name')
        
        relationships_df = relationships_df.merge(
            entity_type_map,
            left_on='source_entity',
            right_on='entity_name',
            how='left'
        ).rename(columns={'entity_type': 'source_entity_type'}).drop(columns=['entity_name'])
        
        relationships_df = relationships_df.merge(
            entity_type_map,
            left_on='target_entity',
            right_on='entity_name',
            how='left'
        ).rename(columns={'entity_type': 'target_entity_type'}).drop(columns=['entity_name'])
        
        return entities_df, relationships_df
    
    def save_graph_data(self, entities_df: pd.DataFrame, relationships_df: pd.DataFrame, version: str = "v1") -> None:
        """
        Save graph data to CSV files.
        
        Args:
            entities_df: Entities DataFrame
            relationships_df: Relationships DataFrame
            version: Version suffix for filenames
        """
        entities_df.to_csv(f'{self.config.graph_data_folder}entities_{version}.csv', index=False)
        relationships_df.to_csv(f'{self.config.graph_data_folder}relationships_{version}.csv', index=False)
    
    def save_graph_output(self, output: List[str], version: str = "v1") -> str:
        """
        Save raw graph output to pickle file.
        
        Args:
            output: List of raw outputs
            version: Version suffix for filename
            
        Returns:
            Path to saved file
        """
        output_file = self.config.get_graph_output_filename(version)
        with open(output_file, 'wb') as f:
            pickle.dump(output, f)
        return output_file
    
    def load_graph_output(self, version: str = "v1") -> List[str]:
        """
        Load raw graph output from pickle file.
        
        Args:
            version: Version suffix for filename
            
        Returns:
            List of raw outputs
        """
        output_file = self.config.get_graph_output_filename(version)
        with open(output_file, 'rb') as f:
            return pickle.load(f) 