"""
Azure OpenAI client management for GraphRAG system.
"""
import time
from typing import List, Optional, Dict, Any
from openai import AzureOpenAI
from .config import Config

class AzureClient:
    """Manages Azure OpenAI client for chat completions and embeddings."""
    
    def __init__(self, config: Config):
        """Initialize Azure client with configuration."""
        self.config = config
        self.chat_client = self._create_chat_client()
        self.embedding_client = self._create_embedding_client()
    
    def _create_chat_client(self) -> AzureOpenAI:
        """Create Azure OpenAI client for chat completions."""
        return AzureOpenAI(
            api_version=self.config.azure.api_version,
            azure_endpoint=self.config.azure.endpoint,
            api_key=self.config.openai_api_key,
        )
    
    def _create_embedding_client(self) -> AzureOpenAI:
        """Create Azure OpenAI client for embeddings."""
        return AzureOpenAI(
            api_version=self.config.azure.embedding_api_version,
            azure_endpoint=self.config.azure.endpoint,
            api_key=self.config.openai_api_key,
        )
    
    def get_chat_response(self, 
                         prompt_content: str, 
                         model: Optional[str] = None,
                         max_retries: Optional[int] = None,
                         retry_delay: Optional[int] = None) -> Dict[str, Any]:
        """
        Get chat completion response with retry logic.
        
        Args:
            prompt_content: The prompt to send to the model
            model: Model to use (defaults to config deployment)
            max_retries: Maximum number of retries (defaults to config)
            retry_delay: Delay between retries in seconds (defaults to config)
            
        Returns:
            Response from Azure OpenAI
            
        Raises:
            Exception: If all retry attempts fail
        """
        model = model or self.config.azure.deployment
        max_retries = max_retries or self.config.processing.max_retries
        retry_delay = retry_delay or self.config.processing.retry_delay
        
        for attempt in range(max_retries):
            try:
                response = self.chat_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system", 
                            "content": self.config.azure.system_prompt
                        },
                        {
                            "role": "user", 
                            "content": prompt_content
                        }
                    ],
                )
                return response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        raise Exception("All attempts to get a response failed.")
    
    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Get embedding for a text string.
        
        Args:
            text: Text to embed
            model: Model to use (defaults to config embedding deployment)
            
        Returns:
            List of embedding values
        """
        model = model or self.config.azure.embedding_deployment
        
        response = self.embedding_client.embeddings.create(
            input=text,
            model=model
        )
        
        return [r.embedding for r in response.data][0]
        
    
    def get_embeddings_batch(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            model: Model to use (defaults to config embedding deployment)
            
        Returns:
            List of embedding vectors
        """
        model = model or self.config.azure.embedding_deployment
        
        response = self.embedding_client.embeddings.create(
            input=texts,
            model=model
        )
        
        return [r.embedding for r in response.data] 