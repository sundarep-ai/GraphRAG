"""
Main script demonstrating GraphRAG usage.
"""
import os
import sys
from pathlib import Path

# Add the code directory to the path
sys.path.append(str(Path(__file__).parent))

from graph_rag import GraphRAG
from config import Config

def main():
    """Main function demonstrating GraphRAG usage."""
    
    # Initialize GraphRAG system
    print("Initializing GraphRAG system...")
    graph_rag = GraphRAG()
    
    # Example 1: Run complete pipeline
    print("\n" + "="*50)
    print("EXAMPLE 1: Complete Pipeline")
    print("="*50)
    
    # You would need to provide the actual file paths
    card_mapping_file = "../data/cardmapping.csv"  # Update with actual path
    chunks_file = "../data/chunks_75000_500.csv"   # Update with actual path or None to process PDFs
    
    try:
        # Check if files exist
        if not os.path.exists(card_mapping_file):
            print(f"Warning: Card mapping file not found: {card_mapping_file}")
            print("Skipping complete pipeline example...")
        else:
            # Run complete pipeline
            results = graph_rag.run_complete_pipeline(
                card_mapping_file=card_mapping_file,
                chunks_file=chunks_file,  # Set to None to process PDFs
                version="v4"
            )
            
            # Print results summary
            print("\nResults Summary:")
            print(f"- Chunks file: {results['chunks_file']}")
            print(f"- Entities: {len(results['entities_df'])}")
            print(f"- Relationships: {len(results['relationships_df'])}")
            print(f"- Embeddings file: {results['embeddings_file']}")
            
            # Get statistics
            entity_stats = graph_rag.get_entity_statistics(results['entities_df'])
            relationship_stats = graph_rag.get_relationship_statistics(results['relationships_df'])
            
            print(f"\nEntity Statistics:")
            print(f"- Total entities: {entity_stats['total_entities']}")
            print(f"- Unique entities: {entity_stats['unique_entities']}")
            print(f"- Cards covered: {entity_stats['cards_covered']}")
            
            print(f"\nRelationship Statistics:")
            print(f"- Total relationships: {relationship_stats['total_relationships']}")
            print(f"- Cards covered: {relationship_stats['cards_covered']}")
            
            # Export results
            graph_rag.export_results(
                results['entities_df'],
                results['relationships_df'],
                results['similarity_pairs'],
                output_dir="results"
            )
    
    except Exception as e:
        print(f"Error in complete pipeline: {e}")
    
    # Example 2: Individual steps
    print("\n" + "="*50)
    print("EXAMPLE 2: Individual Steps")
    print("="*50)
    
    try:
        # Load existing chunks
        if os.path.exists(chunks_file):
            print(f"Loading chunks from: {chunks_file}")
            chunks_df = graph_rag.load_chunks(chunks_file)
            print(f"Loaded {len(chunks_df)} chunks")
            
            # Extract graph data
            print("Extracting graph data...")
            entities_df, relationships_df = graph_rag.extract_graph_data(chunks_file, "v4")
            
            # Generate embeddings
            print("Generating embeddings...")
            relationships_file = f'{graph_rag.config.graph_data_folder}relationships_v4.csv'
            if os.path.exists(relationships_file):
                embeddings_file = graph_rag.generate_embeddings(relationships_file, "v4")
                
                # Analyze similarities
                print("Analyzing similarities...")
                similarity_pairs = graph_rag.analyze_similarities(
                    relationships_file, 
                    embeddings_file,
                    thresholds=[0.85, 0.90, 0.95]
                )
                
                # Print similarity statistics
                for threshold, card_pairs in similarity_pairs.items():
                    total_pairs = sum(len(pairs) for pairs in card_pairs.values())
                    print(f"Threshold {threshold}: {total_pairs} total pairs")
        
        else:
            print(f"Chunks file not found: {chunks_file}")
            print("Skipping individual steps example...")
    
    except Exception as e:
        print(f"Error in individual steps: {e}")
    
    # Example 3: Similarity comparison
    print("\n" + "="*50)
    print("EXAMPLE 3: Similarity Comparison")
    print("="*50)
    
    try:
        # Example relationship descriptions
        text_1 = "The Chase Sapphire Preferred Card offers 2x points on travel and dining"
        text_2 = "Chase Sapphire Preferred earns 2 points per dollar on travel and dining purchases"
        
        print(f"Comparing relationships:")
        print(f"Text 1: {text_1}")
        print(f"Text 2: {text_2}")
        
        is_similar = graph_rag.compare_relationship_similarity(text_1, text_2)
        print(f"Similar: {is_similar}")
        
        # Another example
        text_3 = "Earn 3 points per dollar on groceries for the first 3 months"
        text_4 = "Earn 1 point per dollar on groceries after the initial 3-month period"
        
        print(f"\nComparing relationships:")
        print(f"Text 1: {text_3}")
        print(f"Text 2: {text_4}")
        
        is_similar_2 = graph_rag.compare_relationship_similarity(text_3, text_4)
        print(f"Similar: {is_similar_2}")
    
    except Exception as e:
        print(f"Error in similarity comparison: {e}")
    
    print("\n" + "="*50)
    print("GraphRAG Demo Completed")
    print("="*50)

if __name__ == "__main__":
    main() 