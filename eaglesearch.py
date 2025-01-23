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

class EagleSearch:
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
            with_vectors= True,
            using="original"
        )
        # return response.points
        n=1
        payload = []

        for hit in response.points:
            hit.payload["score"] = hit.score
            payload.append(hit.payload)
    
        return payload

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
