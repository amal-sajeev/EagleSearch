import base64
import csv
import hashlib
import io
import json
import logging
import os
import re
import uuid
import xml.etree.ElementTree as ET
import zipfile

import ebooklib
from ebooklib import epub

import markdown
from bs4 import BeautifulSoup

import nltk
import numpy as np
import pandas as pd
import torch

from PIL import Image

import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import fitz  # PyMuPDF for PDF processing

from colpali_engine.models import ColQwen2, ColQwen2Processor
from qdrant_client import QdrantClient, models

from typing import List, Dict, Any, Union

class EagleSearch:
       
    def __init__(self, 
                 max_chunk_size: int = 500, 
                 similarity_threshold: float = 0.3,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 batch_size: int = 32,
                 max_cache_size: int = 10000):
        """
        Comprehensive document chunking utility supporting multiple file formats.
        
        Args:
            max_chunk_size (int): Maximum characters per chunk
            similarity_threshold (float): Semantic similarity threshold
            embedding_model (str): Sentence embedding model to use
            batch_size (int): Batch size for embedding computation
            max_cache_size (int): Maximum number of cached embeddings
        """
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize embedding model
        self.model = sentence_transformers.SentenceTransformer(embedding_model).to(self.device)
        
        # Chunking parameters
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        
        # Embedding cache management
        self.embedding_cache = {}
        self.max_cache_size = max_cache_size
        
        # Precompile regex for efficiency
        self._whitespace_pattern = re.compile(r'\s+')
        
        # Logging configuration
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def _hash_sentence(self, sentence: str) -> str:
        """Create a stable hash for sentence caching."""
        normalized = self._whitespace_pattern.sub(' ', sentence.strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _cached_embed(self, sentences: List[str]) -> np.ndarray:
        """Cached embedding with batch processing and GPU acceleration."""
        # (Previous implementation remains the same)
        cache_keys = [self._hash_sentence(s) for s in sentences]
        cached_embeddings = [self.embedding_cache.get(key) for key in cache_keys]
        
        needs_embedding = [
            (i, s) for i, (s, emb) in enumerate(zip(sentences, cached_embeddings)) 
            if emb is None
        ]
        
        if needs_embedding:
            batch_sentences = [s for _, s in needs_embedding]
            batch_embeddings = self.model.encode(
                batch_sentences, 
                batch_size=self.batch_size, 
                convert_to_numpy=True,
                device=self.device
            )
            
            for (_, sentence), embedding in zip(needs_embedding, batch_embeddings):
                key = self._hash_sentence(sentence)
                self.embedding_cache[key] = embedding
                
                if len(self.embedding_cache) > self.max_cache_size:
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]
        
        return np.array([
            self.embedding_cache[key] if key in self.embedding_cache else None 
            for key in cache_keys
        ])
    
    def _extract_text_from_markdown(self, file_path: str) -> str:
        """
        Extract plain text from Markdown file.
        
        Args:
            file_path (str): Path to the Markdown file
        
        Returns:
            str: Extracted plain text
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            # Convert Markdown to HTML, then extract text
            md_text = file.read()
            html = markdown.markdown(md_text)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
    
    def _extract_text_from_csv(self, file_path: str) -> str:
        """
        Extract text from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
        
        Returns:
            str: Extracted text from CSV
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Convert all columns to string and concatenate
            text_content = ' '.join(df.apply(lambda row: ' '.join(row.astype(str)), axis=1))
            
            return text_content
        except Exception as e:
            self.logger.error(f"Error processing CSV: {e}")
            return ""
    
    def _extract_text_from_json(self, file_path: str) -> str:
        """
        Extract text from JSON file.
        
        Args:
            file_path (str): Path to the JSON file
        
        Returns:
            str: Extracted text from JSON
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Recursively extract text from nested structures
                def extract_text_from_json(obj):
                    if isinstance(obj, dict):
                        return ' '.join(extract_text_from_json(v) for v in obj.values())
                    elif isinstance(obj, list):
                        return ' '.join(extract_text_from_json(item) for item in obj)
                    else:
                        return str(obj)
                
                return extract_text_from_json(data)
        except Exception as e:
            self.logger.error(f"Error processing JSON: {e}")
            return ""
    
    def _extract_text_from_html(self, file_path: str) -> str:
        """
        Extract text from HTML file.
        
        Args:
            file_path (str): Path to the HTML file
        
        Returns:
            str: Extracted plain text
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            return soup.get_text()
    
    def _extract_text_from_xml(self, file_path: str) -> str:
        """
        Extract text from XML file.
        
        Args:
            file_path (str): Path to the XML file
        
        Returns:
            str: Extracted plain text
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            def extract_text(element):
                text = element.text or ''
                for child in element:
                    text += extract_text(child)
                return text
            
            return extract_text(root)
        except Exception as e:
            self.logger.error(f"Error processing XML: {e}")
            return ""
    
    def _extract_text_from_epub(self, file_path: str) -> str:
        """
        Extract text from EPUB file.
        
        Args:
            file_path (str): Path to the EPUB file
        
        Returns:
            str: Extracted text from EPUB
        """
        try:
            book = epub.read_epub(file_path)
            text_content = []
            
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text_content.append(soup.get_text())
            
            return ' '.join(text_content)
        except Exception as e:
            self.logger.error(f"Error processing EPUB: {e}")
            return ""
    
    def _extract_text_from_log(self, file_path: str) -> str:
        """
        Extract text from log file.
        
        Args:
            file_path (str): Path to the log file
        
        Returns:
            str: Extracted text from log file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Error processing log file: {e}")
            return ""
    
    def _split_long_sentences(self, 
                               sentences: List[str], 
                               max_size: int) -> List[str]:
        """
        Intelligently split sentences longer than chunk size.
        
        Args:
            sentences (List[str]): Input sentences
            max_size (int): Maximum chunk size
        
        Returns:
            List of sentence chunks
        """
        # (Previous implementation remains the same)
        processed_sentences = []
        
        for sentence in sentences:
            if len(sentence) <= max_size:
                processed_sentences.append(sentence)
            else:
                words = sentence.split()
                current_chunk = []
                current_chunk_size = 0
                
                for word in words:
                    if current_chunk_size + len(word) + 1 > max_size:
                        processed_sentences.append(' '.join(current_chunk))
                        current_chunk = []
                        current_chunk_size = 0
                    
                    current_chunk.append(word)
                    current_chunk_size += len(word) + 1
                
                if current_chunk:
                    processed_sentences.append(' '.join(current_chunk))
        
        return processed_sentences
    
    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Universal document chunking method supporting multiple file types.
        
        Args:
            file_path (str): Path to the document file
        
        Returns:
            List of chunk dictionaries with rich metadata
        """
        # Determine file type and extract text
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Text extraction mapping
        extraction_methods = {
            '.txt': lambda path: open(path, 'r', encoding='utf-8').read(),
            '.docx': self._extract_text_from_docx,
            '.md': self._extract_text_from_markdown,
            '.csv': self._extract_text_from_csv,
            '.json': self._extract_text_from_json,
            '.html': self._extract_text_from_html,
            '.xml': self._extract_text_from_xml,
            '.epub': self._extract_text_from_epub,
            '.log': self._extract_text_from_log
        }
        
        # Extract text using appropriate method
        try:
            text_extractor = extraction_methods.get(file_extension)
            if text_extractor:
                full_text = text_extractor(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return []
        
        # Tokenize sentences
        sentences = nltk.sent_tokenize(full_text)
        
        # Handle long sentences
        sentences = self._split_long_sentences(sentences, self.max_chunk_size)
        
        # Compute embeddings
        sentence_embeddings = self._cached_embed(sentences)
        
        # Chunking logic (same as previous implementation)
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        current_chunk_embeddings = []
        
        for sentence, embedding in zip(sentences, sentence_embeddings):
            # Size and semantic coherence checks
            if (current_chunk_size + len(sentence) > self.max_chunk_size or 
                (current_chunk and self._check_semantic_drift(
                    embedding, current_chunk_embeddings))):
                
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'length': len(chunk_text),
                    'embedding': np.mean(current_chunk_embeddings, axis=0) 
                        if current_chunk_embeddings else None
                })
                
                # Reset chunk
                current_chunk = []
                current_chunk_size = 0
                current_chunk_embeddings = []
            
            # Add current sentence
            current_chunk.append(sentence)
            current_chunk_size += len(sentence)
            current_chunk_embeddings.append(embedding)
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'length': len(chunk_text),
                'embedding': np.mean(current_chunk_embeddings, axis=0) 
                    if current_chunk_embeddings else None
            })
        
        return chunks

# Example usage
def main():
    chunker = ComprehensiveDocumentChunker(
        max_chunk_size=500,
        similarity_threshold=0.3
    )
    
    # List of file types to process
    file_types = [
        'document.txt', 
        'document.docx', 
        'document.md', 
        'document.csv', 
        'document.json', 
        'document.html', 
        'document.xml', 
        'document.epub', 
        'document.log'
    ]
    
    # Process each file type
    for file_type in file_types:
        try:
            chunks = chunker.chunk_document(file_type)
            print(f"\n{file_type.upper()} Chunks:")
            for i, chunk in enumerate(chunks, 1):
                print(f"Chunk {i}:")
                print(f"Length: {chunk['length']} characters")
                print(f"Text Preview: {chunk['text'][:100]}...")
                print()
        except Exception as e:
            print(f"Error processing {file_type}: {e}")

if __name__ == "__main__":
    main()