# GraphRAG - Generic Graph-Based Retrieval-Augmented Generation System

A comprehensive object-oriented Python system for processing documents, extracting entities and relationships, and performing similarity analysis using Azure OpenAI services. This system is designed to be versatile and can be applied to any domain or use case.

## Overview

GraphRAG is a generic system that can analyze any type of documents by:
1. **Document Processing**: Converting PDF documents to text chunks
2. **Entity Extraction**: Identifying domain-specific entities and concepts
3. **Relationship Analysis**: Discovering relationships between entities
4. **Similarity Analysis**: Finding similar relationships across different documents
5. **Embedding Generation**: Creating vector representations for semantic search

### Example Use Case: Credit Card Benefits Analysis

The system includes a complete example implementation for analyzing credit card benefit documents, demonstrating how to:
- Extract credit card entities (rewards, fees, insurance, etc.)
- Identify relationships between benefits and card features
- Compare similar benefits across different cards
- Generate insights for card comparison and selection

## Architecture

The system is built using object-oriented programming principles with the following main components:

### Core Classes

- **`Config`**: Centralized configuration management
- **`AzureClient`**: Azure OpenAI API client management
- **`DocumentProcessor`**: PDF processing and text chunking
- **`GraphProcessor`**: Entity and relationship extraction
- **`EmbeddingProcessor`**: Embedding generation and similarity analysis
- **`GraphRAG`**: Main orchestrator class

### Data Classes

- **`TextChunk`**: Represents a text chunk with metadata
- **`Entity`**: Represents an extracted entity
- **`Relationship`**: Represents an extracted relationship
- **`SimilarityPair`**: Represents a pair of similar relationships

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GraphRAG/code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your Azure OpenAI credentials
echo "OpenAI_API_KEY=your_api_key_here" > .env
```

## Configuration

The system uses a centralized configuration system that can be customized for any domain:

### Azure OpenAI Settings
- Endpoint URL
- Model names (chat and embedding)
- API versions
- Retry settings

### Processing Settings
- Chunk size and overlap
- Similarity thresholds
- Entity type definitions

### File Paths
- PDF folder location
- Data output directories
- Document mapping file

## Usage

### Basic Usage

```python
from graph_rag import GraphRAG

# Initialize the system
graph_rag = GraphRAG()

# Run complete pipeline
results = graph_rag.run_complete_pipeline(
    document_mapping_file="data/document_mapping.csv",
    chunks_file=None,  # Set to None to process PDFs
    version="v1"
)
```

### Step-by-Step Usage

```python
# 1. Process documents
chunks_file = graph_rag.process_documents("data/document_mapping.csv")

# 2. Extract graph data
entities_df, relationships_df = graph_rag.extract_graph_data(chunks_file)

# 3. Generate embeddings
embeddings_file = graph_rag.generate_embeddings("data/graph_data/relationships_v1.csv")

# 4. Analyze similarities
similarity_pairs = graph_rag.analyze_similarities(
    "data/graph_data/relationships_v1.csv",
    embeddings_file,
    thresholds=[0.85, 0.90, 0.95]
)
```

### Individual Components

```python
# Document processing
from document_processor import DocumentProcessor
doc_processor = DocumentProcessor(config)
chunks = doc_processor.process_pdf_folder("data/document_mapping.csv")

# Graph processing
from graph_processor import GraphProcessor
graph_processor = GraphProcessor(config, azure_client)
entities, relationships = graph_processor.extract_entities_and_relationships(text_chunks, chunk_index)

# Embedding processing
from embedding_processor import EmbeddingProcessor
embedding_processor = EmbeddingProcessor(config, azure_client)
embeddings = embedding_processor.generate_relationship_embeddings(relationships_df)
```

## Data Flow

1. **Input**: PDF documents in organized folder structure
2. **Processing**: Text extraction, cleaning, and chunking
3. **Extraction**: Entity and relationship identification using LLM
4. **Cleaning**: Entity name normalization and mapping
5. **Embedding**: Vector generation for relationship descriptions
6. **Analysis**: Similarity calculation and clustering
7. **Output**: Structured data files and analysis results

## Output Files

The system generates several output files:

- `chunks_{size}_{overlap}.csv`: Text chunks with metadata
- `entities_{version}.csv`: Extracted entities
- `relationships_{version}.csv`: Extracted relationships
- `relationships_embeddings_{version}.pkl`: Embedding vectors
- `graph_index_output_{version}.pkl`: Raw LLM outputs

## Entity Types

The system uses generic entity types that can be customized for any domain:

### Default Entity Types
- Document, Category, Topic, Concept, Person
- Organization, Location, Date, Amount, Percentage
- Policy, Rule, Benefit, Requirement, Process
- Contact, Terms, Others

### Customizing Entity Types
You can customize entity types in the configuration to match your specific domain:

```python
# In config.py, modify the EntityTypes class
@dataclass
class EntityTypes:
    types: List[str] = None
    
    def __post_init__(self):
        if self.types is None:
            # Customize these for your domain
            self.types = [
                "Product", "Feature", "Specification", "Price",
                "Review", "Rating", "Brand", "Category",
                # Add your domain-specific types here
            ]
```

## Similarity Analysis

The system supports multiple similarity thresholds:
- **0.85**: High similarity (more pairs, lower precision)
- **0.90**: Medium similarity (balanced)
- **0.95**: Very high similarity (fewer pairs, higher precision)

## Performance Considerations

- **GPU Acceleration**: Install CuPy for faster similarity calculations
- **Batch Processing**: The system processes documents in batches
- **Caching**: Embeddings and processed data are cached to avoid reprocessing
- **Retry Logic**: Built-in retry mechanisms for API calls

## Error Handling

The system includes comprehensive error handling:
- API rate limiting and retries
- File I/O error handling
- Data validation and cleaning
- Graceful degradation for missing dependencies

## Customization

### Adding New Entity Types

```python
# In config.py, modify the EntityTypes class
@dataclass
class EntityTypes:
    types: List[str] = None
    
    def __post_init__(self):
        if self.types is None:
            self.types = [
                # ... existing types ...
                "New Entity Type"
            ]
```

### Custom Prompts

```python
# In prompts.py, add new prompt templates
class PromptTemplates:
    @staticmethod
    def get_custom_template() -> str:
        return "Your custom prompt template here"
```

### Custom Processing

```python
# Extend the main classes
class CustomGraphProcessor(GraphProcessor):
    def custom_processing_method(self, data):
        # Your custom processing logic
        pass
```

## Example Implementations

### Credit Card Analysis
The system includes a complete example for credit card benefits analysis:

```python
# Example: Credit card benefits comparison
text_1 = "The Chase Sapphire Preferred Card offers 2x points on travel and dining"
text_2 = "Chase Sapphire Preferred earns 2 points per dollar on travel and dining purchases"

is_similar = graph_rag.compare_relationship_similarity(text_1, text_2)
print(f"Similar: {is_similar}")
```

### Product Comparison
You can adapt the system for product comparison:

```python
# Example: Product feature comparison
text_1 = "iPhone 15 Pro features a 6.1-inch Super Retina XDR display"
text_2 = "The iPhone 15 Pro comes with a 6.1-inch Super Retina XDR screen"

is_similar = graph_rag.compare_relationship_similarity(text_1, text_2)
print(f"Similar: {is_similar}")
```

### Legal Document Analysis
For legal document analysis:

```python
# Example: Legal clause comparison
text_1 = "The contract requires 30 days notice for termination"
text_2 = "Termination of this agreement requires 30 days written notice"

is_similar = graph_rag.compare_relationship_similarity(text_1, text_2)
print(f"Similar: {is_similar}")
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `OpenAI_API_KEY` is set in `.env`
2. **PDF Processing**: Install `docling` for PDF support
3. **GPU Issues**: Install appropriate CuPy version for your CUDA version
4. **Memory Issues**: Reduce chunk size in configuration

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the example scripts

## Acknowledgments

- Azure OpenAI for LLM services
- LangChain for text processing
- Docling for PDF processing
- CuPy for GPU acceleration 