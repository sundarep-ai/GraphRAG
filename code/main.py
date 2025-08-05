"""
Main script demonstrating GraphRAG usage for any domain.
"""
import os
import sys
from pathlib import Path

# Add the code directory to the path
sys.path.append(str(Path(__file__).parent))

from graph_rag import GraphRAG
from config import Config

def main():
    """Main function demonstrating GraphRAG usage for any domain."""
    
    # Initialize GraphRAG system
    print("Initializing GraphRAG system...")
    graph_rag = GraphRAG()
    
    # Example 1: Run complete pipeline
    print("\n" + "="*50)
    print("EXAMPLE 1: Complete Pipeline")
    print("="*50)
    
    # You would need to provide the actual file paths
    document_mapping_file = "../data/document_mapping.csv"  # Update with actual path
    chunks_file = "../data/chunks_75000_500.csv"   # Update with actual path or None to process PDFs
    
    try:
        # Check if files exist
        if not os.path.exists(document_mapping_file):
            print(f"Warning: Document mapping file not found: {document_mapping_file}")
            print("Skipping complete pipeline example...")
        else:
            # Run complete pipeline
            results = graph_rag.run_complete_pipeline(
                document_mapping_file=document_mapping_file,
                chunks_file=chunks_file,  # Set to None to process PDFs
                version="v1"
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
            print(f"- Documents covered: {entity_stats['documents_covered']}")
            
            print(f"\nRelationship Statistics:")
            print(f"- Total relationships: {relationship_stats['total_relationships']}")
            print(f"- Documents covered: {relationship_stats['documents_covered']}")
            
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
            entities_df, relationships_df = graph_rag.extract_graph_data(chunks_file, "v1")
            
            # Generate embeddings
            print("Generating embeddings...")
            relationships_file = f'{graph_rag.config.graph_data_folder}relationships_v1.csv'
            if os.path.exists(relationships_file):
                embeddings_file = graph_rag.generate_embeddings(relationships_file, "v1")
                
                # Analyze similarities
                print("Analyzing similarities...")
                similarity_pairs = graph_rag.analyze_similarities(
                    relationships_file, 
                    embeddings_file,
                    thresholds=[0.85, 0.90, 0.95]
                )
                
                # Print similarity statistics
                for threshold, document_pairs in similarity_pairs.items():
                    total_pairs = sum(len(pairs) for pairs in document_pairs.values())
                    print(f"Threshold {threshold}: {total_pairs} total pairs")
        
        else:
            print(f"Chunks file not found: {chunks_file}")
            print("Skipping individual steps example...")
    
    except Exception as e:
        print(f"Error in individual steps: {e}")
    
    # Example 3: Similarity comparison (Credit Card Example)
    print("\n" + "="*50)
    print("EXAMPLE 3: Similarity Comparison (Credit Card Domain)")
    print("="*50)
    
    try:
        # Example relationship descriptions for credit card domain
        text_1 = "The Chase Sapphire Preferred Card offers 2x points on travel and dining"
        text_2 = "Chase Sapphire Preferred earns 2 points per dollar on travel and dining purchases"
        
        print(f"Comparing relationships (Credit Card Domain):")
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
    
    # Example 4: Product Comparison Example
    print("\n" + "="*50)
    print("EXAMPLE 4: Similarity Comparison (Product Domain)")
    print("="*50)
    
    try:
        # Example relationship descriptions for product domain
        text_1 = "iPhone 15 Pro features a 6.1-inch Super Retina XDR display"
        text_2 = "The iPhone 15 Pro comes with a 6.1-inch Super Retina XDR screen"
        
        print(f"Comparing relationships (Product Domain):")
        print(f"Text 1: {text_1}")
        print(f"Text 2: {text_2}")
        
        is_similar = graph_rag.compare_relationship_similarity(text_1, text_2)
        print(f"Similar: {is_similar}")
        
        # Another example
        text_3 = "The laptop has 16GB RAM and 512GB SSD storage"
        text_4 = "This computer includes 16GB of memory and 512GB solid state drive"
        
        print(f"\nComparing relationships:")
        print(f"Text 1: {text_3}")
        print(f"Text 2: {text_4}")
        
        is_similar_2 = graph_rag.compare_relationship_similarity(text_3, text_4)
        print(f"Similar: {is_similar_2}")
    
    except Exception as e:
        print(f"Error in product similarity comparison: {e}")
    
    # Example 5: Legal Document Example
    print("\n" + "="*50)
    print("EXAMPLE 5: Similarity Comparison (Legal Domain)")
    print("="*50)
    
    try:
        # Example relationship descriptions for legal domain
        text_1 = "The contract requires 30 days notice for termination"
        text_2 = "Termination of this agreement requires 30 days written notice"
        
        print(f"Comparing relationships (Legal Domain):")
        print(f"Text 1: {text_1}")
        print(f"Text 2: {text_2}")
        
        is_similar = graph_rag.compare_relationship_similarity(text_1, text_2)
        print(f"Similar: {is_similar}")
        
        # Another example
        text_3 = "The party shall pay damages for breach of contract"
        text_4 = "Damages must be paid by the party who breaches the agreement"
        
        print(f"\nComparing relationships:")
        print(f"Text 1: {text_3}")
        print(f"Text 2: {text_4}")
        
        is_similar_2 = graph_rag.compare_relationship_similarity(text_3, text_4)
        print(f"Similar: {is_similar_2}")
    
    except Exception as e:
        print(f"Error in legal similarity comparison: {e}")
    
    print("\n" + "="*50)
    print("GraphRAG Demo Completed")
    print("="*50)

if __name__ == "__main__":
    main() 