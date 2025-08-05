"""
Simple example demonstrating GraphRAG usage for any domain.
"""
import os
from graph_rag import GraphRAG

def simple_example():
    """Simple example of using GraphRAG for any domain."""
    
    # Initialize GraphRAG
    print("Initializing GraphRAG system...")
    graph_rag = GraphRAG()
    
    # Example: Compare two relationship descriptions (Credit Card Domain)
    print("\nComparing relationship descriptions (Credit Card Domain):")
    
    text_1 = "The Chase Sapphire Preferred Card offers 2x points on travel and dining"
    text_2 = "Chase Sapphire Preferred earns 2 points per dollar on travel and dining purchases"
    
    print(f"Text 1: {text_1}")
    print(f"Text 2: {text_2}")
    
    try:
        is_similar = graph_rag.compare_relationship_similarity(text_1, text_2)
        print(f"Similar: {is_similar}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set up your OpenAI API key in the .env file")
    
    # Example: Compare two relationship descriptions (Product Domain)
    print("\nComparing relationship descriptions (Product Domain):")
    
    text_3 = "iPhone 15 Pro features a 6.1-inch Super Retina XDR display"
    text_4 = "The iPhone 15 Pro comes with a 6.1-inch Super Retina XDR screen"
    
    print(f"Text 1: {text_3}")
    print(f"Text 2: {text_4}")
    
    try:
        is_similar = graph_rag.compare_relationship_similarity(text_3, text_4)
        print(f"Similar: {is_similar}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example: Compare two relationship descriptions (Legal Domain)
    print("\nComparing relationship descriptions (Legal Domain):")
    
    text_5 = "The contract requires 30 days notice for termination"
    text_6 = "Termination of this agreement requires 30 days written notice"
    
    print(f"Text 1: {text_5}")
    print(f"Text 2: {text_6}")
    
    try:
        is_similar = graph_rag.compare_relationship_similarity(text_5, text_6)
        print(f"Similar: {is_similar}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nExample completed!")

def load_and_analyze_existing_data():
    """Example of loading and analyzing existing data."""
    
    print("Loading and analyzing existing data...")
    graph_rag = GraphRAG()
    
    # Check if data files exist
    chunks_file = "../data/chunks_75000_500.csv"
    relationships_file = "../output_data/graph_data/relationships_v1.csv"
    
    if os.path.exists(chunks_file):
        print(f"Loading chunks from: {chunks_file}")
        chunks_df = graph_rag.load_chunks(chunks_file)
        print(f"Loaded {len(chunks_df)} chunks")
        
        # Show sample data
        print("\nSample chunks:")
        print(chunks_df.head(2))
        
    else:
        print(f"Chunks file not found: {chunks_file}")
    
    if os.path.exists(relationships_file):
        print(f"\nLoading relationships from: {relationships_file}")
        relationships_df = pd.read_csv(relationships_file)
        print(f"Loaded {len(relationships_df)} relationships")
        
        # Show sample data
        print("\nSample relationships:")
        print(relationships_df.head(2))
        
        # Get statistics
        stats = graph_rag.get_relationship_statistics(relationships_df)
        print(f"\nRelationship Statistics:")
        print(f"- Total relationships: {stats['total_relationships']}")
        print(f"- Documents covered: {stats['documents_covered']}")
        print(f"- Strength distribution: {stats['strength_distribution']}")
        
    else:
        print(f"Relationships file not found: {relationships_file}")

if __name__ == "__main__":
    print("GraphRAG Simple Example")
    print("=" * 40)
    
    # Run simple example
    simple_example()
    
    # Try to load existing data
    try:
        import pandas as pd
        load_and_analyze_existing_data()
    except ImportError:
        print("\nPandas not available, skipping data loading example")
    except Exception as e:
        print(f"\nError loading data: {e}")
    
    print("\n" + "=" * 40)
    print("Example completed!") 