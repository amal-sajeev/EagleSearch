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
from io import BytesIO
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
from typing import List, Dict, Any, Union, BinaryIO
from fastapi import UploadFile
import asyncio
import pprint



class EagleSearch:

    def __init__(self,
                 qdrant_api_key:str,
                 qdrant_url : str,
                 max_chunk_size: int = 500,
                 similarity_threshold: float = 0.3,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 batch_size: int = 32,
                 max_cache_size: int = 10000,
                 ):
        """
        Comprehensive document chunking utility supporting multiple file formats.

        Args:
            max_chunk_size (int): Maximum characters per chunkE
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

         # Initialize VLLM model
        self.colmodel = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        ).eval()

        self.e5model = sentence_transformers.SentenceTransformer('intfloat/multilingual-e5-large')

        self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )

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

    def _extract_text_from_markdown(self, file: UploadFile) -> str:
        """
        Extract plain text from Markdown file.
        
        Args:
            file (UploadFile): Uploaded Markdown file
        
        Returns:
            str: Extracted plain text
        """
        content = file.file.read()
        md_text = content.decode("utf-8", errors="replace")
        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()

    def _extract_text_from_csv(self, file: UploadFile) -> str:
        """
        Extract text from CSV file.
        
        Args:
            file (UploadFile): Uploaded CSV file
        
        Returns:
            str: Extracted text from CSV
        """
        try:
            content = file.file.read()
            # Use StringIO to create file-like object for pandas
            df = pd.read_csv(file_path).decode("utf-8", errors="replace")

            # Convert all columns to string and concatenate
            text_content = ' '.join(df.apply(lambda row: ' '.join(row.astype(str)), axis=1))

            return text_content
        except Exception as e:
            self.logger.error(f"Error processing CSV: {e}")
            return ""

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

    def _extract_text_from_docx(self, docx_path: str) -> str:
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
            xml_content = zip_file.read('word/document.xml').decode("utf-8", errors="replace")

            # Efficient XML namespace handling
            namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

            #[EXPERIMENTAL] Using a parser to fix UTF-8 errors
            parser = ET.XMLParser(encoding="UTF-8")

            # Use efficient XML parsing
            tree = ET.fromstring(xml_content, parser = parser)

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
                    text_parts.append(full_para)

            return ' '.join(text_parts)

    def _extract_text_from_json(self,file: UploadFile) -> list[str]:
        """
        Extract text from JSON file with optimized context for LLM retrieval.
        """
        try:
            content = file.file.read()
            data = json.loads(content.decode("utf-8", errors="replace"))

            text_chunks = []
            
            def natural_key(key):
                """Convert camelCase or snake_case to space-separated words"""
                s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', key)
                return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).replace('_', ' ').capitalize()
            
            def process_node(obj, context=None):
                if context is None:
                    context = {}
                
                if isinstance(obj, dict):
                    # Identify important identifier keys
                    identifiers = {}
                    for key in ['name', 'title', 'id', 'type', 'category', 'year', 'releaseYear']:
                        if key in obj and isinstance(obj[key], (str, int, float)):
                            identifiers[key] = obj[key]
                    
                    # Merge with parent context, prioritizing current level's identifiers
                    current_context = {**context, **identifiers}
                    
                    # Process text fields
                    for key, value in obj.items():
                        if isinstance(value, str) and len(value) > 20:
                            # Format meaningful context
                            context_parts = []
                            for ctx_key, ctx_val in current_context.items():
                                if ctx_key not in ['items', 'data', 'children', 'entries']:
                                    context_parts.append(f"{natural_key(ctx_key)}: {ctx_val}")
                            
                            context_str = ", ".join(context_parts)
                            field_name = natural_key(key)
                            
                            if context_str:
                                chunk = f"{context_str}. {field_name}: {value}"
                            else:
                                chunk = f"{field_name}: {value}"
                                
                            text_chunks.append(chunk.encode('utf-8', errors='ignore').decode('utf-8'))
                    
                    # Process nested structures
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            process_node(value, current_context)
                
                elif isinstance(obj, list):
                    for item in obj:
                        process_node(item, context)
            
            process_node(data)
            return [chunk for chunk in text_chunks if chunk.strip()]
        except Exception as e:
            print(f"Error processing JSON: {e}")
            return []
            
    def _extract_text_from_html(self, file: UploadFile) -> str:
        """
        Extract text from HTML file.
        
        Args:
            file (UploadFile): Uploaded HTML file
        
        Returns:
            str: Extracted plain text
        """
        soup = BeautifulSoup(file.file.read().decode("utf-8", errors="replace"), 'html.parser')
        return soup.get_text()

    def _extract_text_from_xml(self, file: UploadFile) -> str:
        """
        Extract text from XML file.
        
        Args:
            file (UploadFile): Uploaded XML file
        
        Returns:
            str: Extracted plain text
        """
        try:
            content = file.file.read()
            root = ET.fromstring(content.decode("utf-8", errors="replace"))

            def extract_text(element):
                text = element.text or ''
                for child in element:
                    text += extract_text(child)
                return text

            return extract_text(root)
        except Exception as e:
            self.logger.error(f"Error processing XML: {e}")
            return ""

    def _extract_text_from_epub(self, file: UploadFile) -> str:
        """
        Extract text from EPUB file.
        
        Args:
            file (UploadFile): Uploaded EPUB file
        
        Returns:
            str: Extracted text from EPUB
        """
        try:
            content = file.file.read()
            epub_bytes = io.BytesIO(content)
            book = epub.read_epub(epub_bytes)
            text_content = []

            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text_content.append(soup.get_text())

            return ' '.join(text_content)
        except Exception as e:
            self.logger.error(f"Error processing EPUB: {e}")
            return ""

    def _extract_text_from_log(self, file: UploadFile) -> str:
        """
        Extract text from log file.
        
        Args:
            file (UploadFile): Uploaded log file
        
        Returns:
            str: Extracted text from log file
        """
        try:
            content = file.file.read()
            text = content.decode("utf-8", errors="replace")
            return text

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

    def _setup_collection(self, collection_name, txt:bool=False):
        """Create Qdrant collection if it doesn't exist"""
        try:
            self.client.get_collection(collection_name)
        except:
            if txt == True:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "txt_vectors": models.VectorParams(
                            size=1024,
                            distance=models.Distance.COSINE,
                            multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM
                            ),
                            hnsw_config=models.HnswConfigDiff(m=0)
                        )
                    }
                )
            else:
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
        image_embeddings = self.colmodel(**processed_image)

        # Get first embedding (batch size 1)
        image_embedding = image_embeddings[0]

        # Identify image tokens
        mask = processed_image.input_ids[0] == self.processor.image_token_id

        # Get patches dimensions
        x_patches, y_patches = self.processor.get_n_patches(
            image.size,
            patch_size=self.colmodel.patch_size,
            spatial_merge_size= self.colmodel.spatial_merge_size
        )

        # Reshape and mean pool
        image_tokens = image_embedding[mask].view(x_patches, y_patches, self.colmodel.dim)
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

    # Embedding and uploading pdf pages
    def _ingest_pdf(self, pdf: Union[str, UploadFile], collection_name="", batch_size=4):
        """Process entire PDF and store vectors with batch processing"""
        self._setup_collection(collection_name)
        if type(pdf) == str:
            doc = fitz.open(pdf)
        else:
            doc = fitz.open(stream = pdf.file, filetype = "pdf")
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
                            "doc_name": doc.name.split("/")[-1],
                            "page_number": page_num,
                            "metadata": metadata,
                            "text_content": text_data,
                            "page_dimensions": {
                                "width": float(page.rect.width),
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
        return(total_pages)

    def _ingest_photos(self, images: Union[List[UploadFile], UploadFile], collection_name: str = ""):
        """Processes and embeds JPEG and PNG files
        Args:
            image (BytesIO): List of Image data as BytesIO
            collection_name (str, optional): _description_. Defaults to "".
        """
        self._setup_collection(collection_name)

        point_batch = []
        if type(images) == UploadFile:
            temp = []
            temp.append(images)
            images = temp
        for itimage in images:
            try:
                #Convert BytesIO image to PIL Image
                image = Image.open(itimage.file)

                #Embed Image
                image_vectors = self._process_page(image)
                #Converting image to standard PNG to Base64 string
                imgbuffer = BytesIO()
                image.save(imgbuffer, format = "PNG")
                # metadata["page_image"] = base64.b64encode(imgbuffer.getvalue()).decode()
                point = models.PointStruct(
                            id = str(uuid.uuid4()),
                            vector = image_vectors,
                            payload = {
                                "doc_id" : str(uuid.uuid4()),
                                "doc_name" : str(itimage.filename.split("/")[-1]),
                                "page_image" : base64.b64encode(imgbuffer.getvalue()).decode(),
                                ""
                                "page_dimensions" : {
                                    "width" : float(image.width),
                                    "height" : float(image.height)
                                }
                            }
                        )
                point_batch.append(point)

            except Exception as e:
                        print(f"Error processing image {itimage.filename}: {str(e)}")
                        continue
        try:
            self.client.upsert(
                collection_name= collection_name,
                points = point_batch
            )
            return(len(images))
        except Exception as e:
                    print(f"Error uploading batch: {str(e)}")
                    # Print the first point's payload for debugging
                    if point_batch:
                        print("First point payload sample:")
                        print(point_batch[0].payload)

    def chunk_document(self, file: UploadFile) -> List[Dict[str, Any]]:
        """
        Universal document chunking method supporting multiple file types.

        Args:
            file (UploadFile): The uploaded document file

        Returns:
            List of chunk dictionaries with rich metadata
        """
        # Determine file type and extract text
        file_extension = os.path.splitext(file.filename)[1].lower()

        # Text extraction mapping
        extraction_methods = {
            '.txt': lambda path: file.file.read().decode("utf-8", errors="replace"),
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
                full_text = text_extractor(file)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return []
        
        # Use balanced chunking approach
        return self._balanced_chunking(full_text, file_extension)

    def _balanced_chunking(self, full_text, file_extension) -> List[Dict[str, Any]]:
        """
        Creates balanced chunks that preserve context while maintaining reasonable size.
        Target chunk size is between 800-1200 characters.
        """
        # Define target chunk size range
        MIN_CHUNK_SIZE = 600  # Characters
        IDEAL_CHUNK_SIZE = 1000  # Characters
        MAX_CHUNK_SIZE = 1500  # Characters
        
        if file_extension == ".json":
            # Handle JSON content as before
            sentences= full_text
        else:
            # Split into sentences
            sentences = nltk.sent_tokenize(str(full_text))
        
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # Handle exceptionally long sentences
            if sentence_len > MAX_CHUNK_SIZE:
                # If we have content in the current chunk, add it first
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'length': len(chunk_text)
                    })
                    current_chunk = []
                    current_chunk_size = 0
                
                # Split the long sentence at reasonable points (e.g., punctuation, spaces)
                split_points = [match.start() for match in re.finditer(r'[.!?:;,] ', sentence)]
                split_points = [p for p in split_points if p > MIN_CHUNK_SIZE and p < MAX_CHUNK_SIZE] + [len(sentence)]
                
                start = 0
                for end in split_points:
                    if end - start > MIN_CHUNK_SIZE or end == split_points[-1]:
                        chunks.append({
                            'text': sentence[start:end].strip(),
                            'length': end - start
                        })
                        start = end + 1
                        
                        # Break if we've processed the whole sentence
                        if start >= len(sentence):
                            break
                
                continue
            
            # For normal-sized sentences
            if current_chunk_size + sentence_len > IDEAL_CHUNK_SIZE:
                # If adding this sentence would exceed MAX_CHUNK_SIZE or we're already above IDEAL_CHUNK_SIZE
                if (current_chunk_size + sentence_len > MAX_CHUNK_SIZE or 
                    current_chunk_size >= IDEAL_CHUNK_SIZE):
                    # Create a chunk with current content
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append({
                            'text': chunk_text,
                            'length': len(chunk_text)
                        })
                        current_chunk = []
                        current_chunk_size = 0
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_chunk_size += sentence_len
            
            # Add spacing between sentences in the size calculation
            if len(current_chunk) > 1:
                current_chunk_size += 1  # For the space between sentences
        
        # Add the final chunk if there's content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'length': len(chunk_text)
            })
        
        return chunks

    def _ingest_text(self, chunks: List[str], file:UploadFile, collection_name: str = "", client_id = "", bot_id = "", batch_size: str = 4):
        """Process chunked text into embedded vectors with ColQwen and upload the vectors.

        Args:
            text (List): Text chunks to embed.
            batch_size (int, optional): Batch size for parallel uploading. Defaults to 4.
        """
        self._setup_collection(collection_name, True)

        # Generate embeddings for text chunks
        embeddings = []
        #E5 embedding
        pprint.pprint(chunks)
        proembeddings = self.e5model.encode([i["text"] for i in chunks],prompt="passage:")
        for chunk,embed in zip(chunks,proembeddings):
            embeddings.append((str(uuid.uuid4()),embed,chunk))
        
        # Get embedding dimension
        embedding_dimension = embeddings[0][1].shape[0]

        print(embedding_dimension)

        # Prepare points for uploading
        points = []
        doc_id = str(uuid.uuid4())
        for idx, embedding, text in embeddings:
            points.append(
                models.PointStruct(
                    id=idx,
                    vector = {"txt_vectors": embedding},
                    payload={
                        "doc_id": doc_id,
                        "doc_name": file.filename,
                        "client": client_id,  # Add client_id for search filtering
                        "bot_id": bot_id,     # Add bot_id for search filtering
                        "type": file.filename.split(".")[-1],        # Add type identifier
                        "content": text
                    }
                )
            )
        
        # Upload points to Qdrant
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        return len(points)
            
    #   Main Ingestion function
    def ingest(self, client_id, bot_id, files: Union[UploadFile,List[UploadFile]], txt_collection:str = "", img_collection:str = ""):
        """Chunk content of files, embed them into vectors, and upload them to vector store.
    
        Args:
            files (Union[str,BytesIO,List[str],List[UploadFile]]): File or list of Files to ingest.
            txt_collection (str, optional): The collection that all text files will be uploaded to. Defaults to "".
            img_collection (str, optional): The collection that all image, presentation, and pdf files will be uploaded to. Defaults to "".
        """
        imgformats = ["png","jpg","pdf"]
        txformats = ['txt','docx','md','csv','json','html','xml','epub','log']

        flist = [] 
        if type(files) == UploadFile:
            flist.append(files)
        else:
            flist = files

        if txt_collection !="":
            self._setup_collection(txt_collection,txt=True)
        if img_collection !="":
            self._setup_collection(img_collection)

        for i in flist:
            
            fformat = i.filename.split(".")[-1]
            print(fformat)
            if fformat in txformats:
                if txt_collection != "":
                    txchunks = self.chunk_document(i)
                    self._ingest_text(
                        chunks =txchunks,
                        file = i,
                        collection_name= txt_collection,
                        client_id= client_id,
                        bot_id = bot_id,
                    batch_size=5)
            elif fformat in imgformats:
                if img_collection != "":
                    
                    if fformat == "pdf":
                        print("pdf check passed")
                        self._ingest_pdf(
                            pdf=i,
                            collection_name= img_collection,
                            batch_size=5)
                    else:
                        print("png check passed")
                        self._ingest_photos(
                            images = i,
                            collection_name=img_collection
                            )
            else:
                raise ValueError(f"Unsupported file type: {fformat}")

    def search(self, query, limit=10, prefetch_limit=100, client_id = "", bot_id ="", txt_collection:str = "", img_collection:str=""):
        """Retuns an array of strings of image data and an array of text of the matching pages.

        Args:
            query (_type_): The text content of the query.
            limit (int, optional): Number of results to return. Defaults to 10.
            prefetch_limit (int, optional): Number of results to fetch from the compressed vector data before reranking. Higher means slower. Defaults to 100.

        Returns:
            img_array: array of pdf pages
            text_array: array of text extracts as strings
        """

        #E5 Query embed
        txtquery_embedding = self.e5model.encode(query,prompt="query:")

        #Qwen Query embed
        processed_query = self.processor.process_queries([query]).to(self.model.device)
        query_embedding = self.colmodel(**processed_query)[0]
        query_embedding = query_embedding.to(torch.float32).detach().cpu().numpy()

        n=1
        payload = []


        #crafting the query filter for client and bot
        # query
        if txt_collection != "":
            text_response = self.client.query_points(
                collection_name = txt_collection,
                query = txtquery_embedding,
                query_filter= models.Filter(
                    must=[
                        models.FieldCondition(
                            key = "client",
                            match = models.MatchValue(
                                value= client_id
                            )
                        ),
                        models.FieldCondition(
                            key = "bot_id",
                            match = models.MatchValue(
                                value = bot_id
                            )
                        )
                    ]
                ),
                using = "txt_vectors",
                limit=limit
            )
            for hit in text_response.points:
                hit.payload["score"] = hit.score
                payload.append(hit.payload)

        if img_collection != "":
            img_response = self.client.query_points(
                collection_name=img_collection,
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
            for hit in img_response.points:
                hit.payload["score"] = hit.score
                payload.append(hit.payload)

        # return response.points
        return payload

    #Search by document_id
    def search_by_docid(self, doc_id:str, collection_name:str):
        """Get the ids of points in a collection that have chunks from a document.

        Args:
            doc_id (str): UUID of the document
            collection_name (str): Collection where all the vectors belonging to the document are stored.

        Returns:
            List[str]: A list of all point IDs. 
        """
        pointlist=[]
        pointnum = 0
        offset = None
        while True:
            points, offset = self.client.scroll(
            collection_name,
            scroll_filter= models.Filter(
                must = [
                    models.FieldCondition(
                        key="doc_id",
                        match = models.MatchValue(value=doc_id)
                    )
                ]
            ),
            offset=offset
            )
            
            pointlist.extend(points)
            if not offset:
                break
            pointnum += len(points)
            print(pointnum)
            print(offset)
        
        return(pointlist)

    #Delete vectors (EXPERIMENTAL)
    def delete_by_docid(self, doc_id:str, collection_name:str):
        """Delete all points that contain a document ID from a collection.

        Args:
            doc_id (str): ID of the document to filter by.
            collection_name (str): Collection to delete points from.

        Returns:
            int: Number of deleted points.
        """

        deleted = self.client.delete(
            collection_name,
            points_selector = models.Filter(
                must = [
                    models.FieldCondition(
                        key="doc_id",
                        match = models.MatchValue(value=doc_id)
                    )
                ]
            ))
        return(deleted)



    # BYTE TO IMAGE CONVERSION FUNCTIONS ======================================================================

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
 