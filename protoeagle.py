import fitz  # PyMuPDF for PDF processing
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from qdrant_client import QdrantClient, models
from PIL import Image
import numpy as np
import io
import json
import base64
from io import BytesIO
import uuid
import tqdm
from typing import Union, List
import numpy as np
from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer

class EagleSearch:

    # VLLM EMBEDDING AND PDF SEARCH FUNCTIONS =====================================================================


    def __init__(self, qdrant_url, qdrant_api_key):
        # Initialize VLLM model
        self.model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        ).eval()
        
        self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )

    def _setup_collection(self, collection_name):
        """Create Qdrant collection if it doesn't exist"""
        try:
            self.client.get_collection(collection_name)
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "original": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0)  # Disable HNSW for original vectors
                    ),
                    "mean_pooling_columns": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    ),
                    "mean_pooling_rows": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    )
                }
            )

    def _clean_text_data(self, text_data):
        """Clean text data to ensure it's JSON serializable"""
        if isinstance(text_data, str):
            # Replace or remove invalid characters
            return text_data.encode('utf-8', errors='ignore').decode('utf-8')
        elif isinstance(text_data, dict):
            return {k: self._clean_text_data(v) for k, v in text_data.items()}
        elif isinstance(text_data, list):
            return [self._clean_text_data(item) for item in text_data]
        else:
            return text_data

    def _extract_page_text(self, page):
        """Extract text from a PDF page with encoding error handling"""
        try:

            # Get text with HTML formatting
            text_html = page.get_text("html")

            # Get plain text
            text_plain = page.get_text("text", sort=True)
            
            # Get text blocks with position information
            blocks = page.get_text("blocks")
            structured_blocks = []
            
            for b in blocks:
                # Each block contains: (x0, y0, x1, y1, "text", block_no, block_type)
                structured_blocks.append({
                    "bbox": (b[0], b[1], b[2], b[3]),
                    "text": b[4],
                    "block_no": b[5],
                    "block_type": b[6]  # 0=text, 1=image, etc.
                })
            
            # Get text in JSON format with detailed layout info
            text_dict = page.get_text("dict")

            text_data = {
            "text_plain": text_plain,
            "text_html": text_html,
            "blocks": structured_blocks
        }
            
            # Clean the text data
            return(self._clean_text_data(text_data))
            
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return {"text_plain": "", "blocks": []}
            
    def _convert_page_to_image(self, page):
        """Alternative method to convert PDF page to PIL Image"""
        try:
            # First attempt - direct conversion
            zoom = 2
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img
        except Exception as e:
            print(f"Direct conversion failed: {e}")
            try:
                # Second attempt - using temporary PNG
                zoom = 2
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix)
                
                # Save to bytes buffer
                buffer = io.BytesIO()
                pix.save(buffer, "png")
                buffer.seek(0)
                
                # Open with PIL
                img = Image.open(buffer)
                return img.convert('RGB')
            except Exception as e:
                print(f"Alternative conversion failed: {e}")
                raise ValueError(f"Could not convert page to image: {e}")

    def _process_page(self, image):
        """Generate vectors for a single page"""
        processed_image = self.processor.process_images([image])
        image_embeddings = self.model(**processed_image)
        
        # Get first embedding (batch size 1)
        image_embedding = image_embeddings[0]
        
        # Identify image tokens
        mask = processed_image.input_ids[0] == self.processor.image_token_id
        
        # Get patches dimensions
        x_patches, y_patches = self.processor.get_n_patches(
            image.size,
            patch_size=self.model.patch_size,
            spatial_merge_size= self.model.spatial_merge_size
        )
        
        # Reshape and mean pool
        image_tokens = image_embedding[mask].view(x_patches, y_patches, self.model.dim)
        pooled_by_rows = image_tokens.mean(dim=0)
        pooled_by_columns = image_tokens.mean(dim=1)
        
        # Add special tokens back
        non_image_embeddings = image_embedding[~mask]
        pooled_by_rows = torch.cat([pooled_by_rows, non_image_embeddings])
        pooled_by_columns = torch.cat([pooled_by_columns, non_image_embeddings])
        
        # Convert to float32 before converting to numpy
        return {
            "original": image_embedding.to(torch.float32).detach().cpu().numpy(),
            "mean_pooling_rows": pooled_by_rows.to(torch.float32).detach().cpu().numpy(),
            "mean_pooling_columns": pooled_by_columns.to(torch.float32).detach().cpu().numpy()
        }

        
    def ingest_pdf(self, pdf: Union[str,BytesIO], collection_name="", batch_size=4):
        """Process entire PDF and store vectors with batch processing"""
        self._setup_collection(collection_name)
        if type(pdf) == type("haha"):
            doc = fitz.open(pdf)
        else:
            doc = fitz.open(stream = pdf.read(), filetype = "pdf")
        metadata = doc.metadata
        
        # Clean metadata
        if metadata:
            metadata = self._clean_text_data(metadata)
        
        total_pages = len(doc)
        
        doc_id = str(uuid.uuid4())

        # Process pages in batches
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_points = []
            
            # Process each page in the current batch
            for page_num in range(batch_start, batch_end):
                page = doc[page_num]
                
                try:
                    # Extract and clean text
                    text_data = self._extract_page_text(page)
                    # Convert to image and process
                    image = self._convert_page_to_image(page)
                    vectors = self._process_page(image)
                    buffered = BytesIO()
                    image.save(buffered,format="PNG")
                    # Create point structure for this page
                    metadata["page_image"] = base64.b64encode(buffered.getvalue()).decode()
                    point = models.PointStruct(
                        id= str(uuid.uuid4()),
                        vector=vectors,
                        payload={
                            "doc_id" : doc_id,
                            "page_id": f"{doc_id}_{page_num}",
                            "pdf_name": doc.name.split("/")[-1],
                            "page_number": page_num,
                            "metadata": metadata,
                            "text_content": text_data,
                            "page_dimensions": {
                                "width": float(page.rect.width),  # Convert to float
                                "height": float(page.rect.height)
                            }
                        }
                    )
                    batch_points.append(point)
                    
                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
                    continue
            
            # Batch upload to Qdrant
            if batch_points:
                try:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=batch_points
                    )
                    print(f"Processed and uploaded pages {batch_start} to {batch_end-1}")
                except Exception as e:
                    print(f"Error uploading batch {batch_start}-{batch_end-1}: {str(e)}")
                    # Print the first point's payload for debugging
                    if batch_points:
                        print("First point payload sample:")
                        print(batch_points[0].payload)
        
        doc.close()

    def ingest_multiple_pdfs(self, pdfs: Union[List[str], List[BytesIO]], batch_size=4):
        """
        Process multiple PDFs sequentially
        Args:
            pdfs: Either a list of paths to PDF files or a list of BytesIO objects
            batch_size: Number of pages to process at once for each PDF
        """
        for pdf in pdfs:
            try:
                print(f"Processing {pdf}")
                self.ingest_pdf(pdf, batch_size)
                print(f"Completed processing {pdf}")
            except Exception as e:
                print(f"Error processing {pdf}: {str(e)}")
                continue

    def search(self, query, limit=10, prefetch_limit=100, collection_name:str=""):
        """Retuns a string of image data of the matching pages.

        Args:
            query (_type_): The text content of the query.
            limit (int, optional): Number of results to return. Defaults to 10.
            prefetch_limit (int, optional): Number of results to fetch from the compressed vector data before reranking. Higher means slower. Defaults to 100.

        Returns:
            _type_: _description_
        """
        self._setup_collection(collection_name)
        processed_query = self.processor.process_queries([query]).to(self.model.device)
        query_embedding = self.model(**processed_query)[0]
        query_embedding = query_embedding.to(torch.float32).detach().cpu().numpy()
        
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            prefetch=[
                models.Prefetch(
                    query=query_embedding,
                    limit=prefetch_limit,
                    using="mean_pooling_columns"
                ),
                models.Prefetch(
                    query=query_embedding,
                    limit=prefetch_limit,
                    using="mean_pooling_rows"
                ),
            ],
            limit=limit,
            with_payload=True,
            using="original"
        )
        # return response.points
        n=1
        payload = []

        for hit in response.points:
            hit.payload["score"] = hit.score
            payload.append(hit.payload)
    
        return payload

    # PAGE TO IMAGE CONVERSION FUNCTIONS ======================================================================

    def base64_to_image(self, base64_string):
        """
        Convert a base64 string to an image file
        
        Parameters:
        base64_string (str): The base64 encoded image string
        
        Returns:
        PIL.Image: The decoded image
        """
        # Remove the data URL prefix if it exists
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode the base64 string
        img_data = base64.b64decode(base64_string)
        
        # Create an image object from the decoded data
        img = Image.open(io.BytesIO(img_data))
        
        return img

    # Example usage:
    def save_image(self,base64_string, output_path):
            """
            Convert base64 string to image and save to file
            
            Parameters:
            base64_string (str): The base64 encoded image string
            output_path (str): Path where the image should be saved
            """
            img = self.base64_to_image(base64_string)
            img.save(output_path)

# TXT CHUNKER FUNCTION ========================================================================================

class EagleSearchTXT:
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2', 
                 batch_size: int = 32,
                 max_cache_size: int = 100000
                 ):
        """
        Optimized semantic chunking with multiple performance improvements.
        
        Args:
            model_name (str): Embedding model to use
            batch_size (int): Batch size for embedding computation
        """
        # Use GPU if available for faster computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model once and move to GPU
        self.model = SentenceTransformer(model_name).to(self.device)
        
        # Configurable parameters
        self.batch_size = batch_size
        
        # Caching mechanism for embeddings
        self.embedding_cache = {}
    
    def _cached_embed(self, sentences: List[str]) -> np.ndarray:
        """
        Cached embedding with batch processing and GPU acceleration.
        
        Cache strategy prevents recomputing embeddings for same sentences.
        Batch processing reduces overhead of embedding computation.
        """
        # Check cache first
        cache_keys = [self._hash_sentence(s) for s in sentences]
        cached_embeddings = [self.embedding_cache.get(key) for key in cache_keys]
        
        # Identify sentences that need embedding
        needs_embedding = [
            (i, s) for i, (s, emb) in enumerate(zip(sentences, cached_embeddings)) 
            if emb is None
        ]
        
        # Batch embed uncached sentences
        if needs_embedding:
            batch_sentences = [s for _, s in needs_embedding]
            batch_embeddings = self.model.encode(
                batch_sentences, 
                batch_size=self.batch_size, 
                convert_to_numpy=True,
                device=self.device
            )
            
            # Update cache with LRU (Least Recently Used) eviction
            for (_, sentence), embedding in zip(needs_embedding, batch_embeddings):
                key = self._hash_sentence(sentence)
                self.embedding_cache[key] = embedding
                
                # Implement LRU cache eviction
                if len(self.embedding_cache) > self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]
        
        # Retrieve final embeddings (from cache or just computed)
        return np.array([
            self.embedding_cache[key] if key in self.embedding_cache else None 
            for key in cache_keys
        ])
    
    def _hash_sentence(self, sentence: str) -> str:
        """
        Create a stable hash for sentence to use as cache key.
        Prevents recomputing embeddings for identical sentences.
        """
        return hash(sentence.strip())
    
    def smart_split_long_sentence(sentence, max_size):
        """
        Split excessively long sentences without breaking words.
        
        Args:
            sentence (str): Input sentence
            max_size (int): Maximum chunk size
        
        Returns:
            List of sentence chunks
        """
        # If sentence is shorter than max size, return as-is
        if len(sentence) <= max_size:
            return [sentence]
        
        # Split strategy: break at word boundaries
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for word in sentence.split():
            # If adding this word exceeds chunk size, start a new chunk
            if current_chunk_size + len(word) + 1 > max_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_chunk_size = 0
            
            current_chunk.append(word)
            current_chunk_size += len(word) + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    

    def chunk_text(self, text: str, max_chunk_size: int = 2000, similarity_threshold: float = 0.4, hardlimit: bool = False) -> List[str]:
        """Divide the content of TXT files into smaller chunks.

        Args:
            text (str): Input long text to chunk.
            hardlimit (bool, optional): Turn on a hard limit . Defaults to False.

        Returns:
            List[str]: _description_
        """
        sentences = self._tokenize_sentences(text)
        sentence_embeddings = self._cached_embed(sentences)
        
        chunks = []
        current_chunk = []
        current_chunk_size = 0

        
        for sentence, embedding in zip(sentences, sentence_embeddings):
            # Compute potential new chunk size
            potential_chunk_size = current_chunk_size + len(sentence)
            
            # Split strategies
            print(f"Chunk now: {current_chunk_size},  size limit: {max_chunk_size}")
            if potential_chunk_size >= max_chunk_size:
                print("splitting!")
                # Current chunk is full, save it
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_chunk_size = len(sentence)
            
            # Optional: Add semantic similarity check
            elif current_chunk and self._compute_max_similarity(embedding, self._cached_embed(''.join(current_chunk))) < similarity_threshold:
                chunks.append(''.join(current_chunk))                     d
                current_chunk = [sentence]
                current_chunk_size = len(sentence)
            
            else:
                # Continue building current chunk
                current_chunk.append(sentence)
                current_chunk_size += len(sentence)
        


        # Add final chunk
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks
        
    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Advanced sentence tokenization with more robust handling.
            Dan be replaced with more sophisticated NLP libraries.
        """
        import nltk
        nltk.download('punkt_tab', quiet=True)
        return nltk.sent_tokenize(text)
    
    def _compute_max_similarity(self, embedding: np.ndarray, 
                                 prev_embeddings: np.ndarray) -> float:
        """
        Efficiently compute maximum cosine similarity.
        Uses numpy for vectorized operations.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Handle empty previous embeddings
        if len(prev_embeddings) == 0:
            return 0.0
        
        # Vectorized similarity computation
        similarities = cosine_similarity([embedding], prev_embeddings)[0]
        return np.max(similarities)
    

import io
import os
import xml.etree.ElementTree as ET
import zipfile
from typing import List, Dict, Any, Union
import hashlib
import re

import torch
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class UniversalDocumentChunker:
    def __init__(self, 
                 max_chunk_size: int = 500, 
                 similarity_threshold: float = 0.3,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 batch_size: int = 32,
                 max_cache_size: int = 10000):
        """
        Comprehensive document chunking utility supporting multiple file types.
        
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
        self.model = SentenceTransformer(embedding_model).to(self.device)
        
        # Chunking parameters
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        
        # Embedding cache management
        self.embedding_cache = {}
        self.max_cache_size = max_cache_size
        
        # Precompile regex for efficiency
        self._whitespace_pattern = re.compile(r'\s+')
    
    def _hash_sentence(self, sentence: str) -> str:
        """
        Create a stable, consistent hash for sentence caching.
        
        Args:
            sentence (str): Input sentence to hash
        
        Returns:
            str: Consistent hash value for the sentence
        """
        # Normalize sentence by removing extra whitespace
        normalized = self._whitespace_pattern.sub(' ', sentence.strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _cached_embed(self, sentences: List[str]) -> np.ndarray:
        """
        Cached embedding with batch processing and GPU acceleration.
        
        Cache strategy prevents recomputing embeddings for same sentences.
        Batch processing reduces overhead of embedding computation.
        """
        # Check cache first
        cache_keys = [self._hash_sentence(s) for s in sentences]
        cached_embeddings = [self.embedding_cache.get(key) for key in cache_keys]
        
        # Identify sentences that need embedding
        needs_embedding = [
            (i, s) for i, (s, emb) in enumerate(zip(sentences, cached_embeddings)) 
            if emb is None
        ]
        
        # Batch embed uncached sentences
        if needs_embedding:
            batch_sentences = [s for _, s in needs_embedding]
            batch_embeddings = self.model.encode(
                batch_sentences, 
                batch_size=self.batch_size, 
                convert_to_numpy=True,
                device=self.device
            )
            
            # Update cache with LRU (Least Recently Used) eviction
            for (_, sentence), embedding in zip(needs_embedding, batch_embeddings):
                key = self._hash_sentence(sentence)
                self.embedding_cache[key] = embedding
                
                # Implement LRU cache eviction
                if len(self.embedding_cache) > self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]
        
        # Retrieve final embeddings (from cache or just computed)
        return np.array([
            self.embedding_cache[key] if key in self.embedding_cache else None 
            for key in cache_keys
        ])
    
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
        processed_sentences = []
        
        for sentence in sentences:
            if len(sentence) <= max_size:
                processed_sentences.append(sentence)
            else:
                # Split at word boundaries
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
                
                # Add final chunk
                if current_chunk:
                    processed_sentences.append(' '.join(current_chunk))
        
        return processed_sentences
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """
        Efficiently extract plain text from a .docx file.
        
        Args:
            docx_path (str): Path to the .docx file
        
        Returns:
            str: Extracted plain text with structural preservation
        """
        # Use context manager for efficient file handling
        with zipfile.ZipFile(docx_path) as zip_file:
            # Directly read XML content
            xml_content = zip_file.read('word/document.xml')
            
            # Efficient XML namespace handling
            namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            # Use efficient XML parsing
            tree = ET.fromstring(xml_content)
            
            # Efficient text extraction
            text_parts = []
            for para in tree.findall('.//w:p', namespaces=namespace):
                para_text = [
                    run.text for run in para.findall('.//w:t', namespaces=namespace) 
                    if run.text
                ]
                
                # Join paragraph text efficiently
                full_para = ' '.join(para_text).strip()
                if full_para:
                    text_parts.append(f"[PARAGRAPH] {full_para}")
            
            return ' '.join(text_parts)
    
    def _check_semantic_drift(self, 
                               current_embedding: np.ndarray, 
                               chunk_embeddings: List[np.ndarray]) -> bool:
        """
        Detect semantic drift using cosine similarity.
        
        Args:
            current_embedding (np.ndarray): Embedding of current sentence
            chunk_embeddings (List[np.ndarray]): Embeddings in current chunk
        
        Returns:
            bool: Whether semantic drift has occurred
        """
        if not chunk_embeddings:
            return False
        
        similarities = cosine_similarity(
            [current_embedding], 
            chunk_embeddings
        )[0]
        
        return np.max(similarities) < self.similarity_threshold
    
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
        
        if file_extension == '.docx':
            full_text = self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                full_text = file.read()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Tokenize sentences
        sentences = nltk.sent_tokenize(full_text)
        
        # Handle long sentences first
        sentences = self._split_long_sentences(sentences, self.max_chunk_size)
        
        # Compute embeddings
        sentence_embeddings = self._cached_embed(sentences)
        
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
    chunker = UniversalDocumentChunker(
        max_chunk_size=500,
        similarity_threshold=0.3
    )
    
    # Process a .txt file
    txt_chunks = chunker.chunk_document('path/to/your/document.txt')
    
    # Process a .docx file
    docx_chunks = chunker.chunk_document('path/to/your/document.docx')
    
    # Print chunks with their metadata
    def print_chunks(chunks, file_type):
        print(f"{file_type.upper()} Document Chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}:")
            print(f"Length: {chunk['length']} characters")
            print(f"Text Preview: {chunk['text'][:100]}...")
            print()
    
    print_chunks(txt_chunks, 'txt')
    print_chunks(docx_chunks, 'docx')

if __name__ == "__main__":
    main()