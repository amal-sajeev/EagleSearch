import fitz  # PyMuPDF for PDF processing
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient, models
from PIL import Image
import numpy as np
import io
import json

class PDFProcessor:
    def __init__(self, qdrant_url, qdrant_api_key, collection_name="pdf_vectors"):
        # Initialize VLLM model
        self.model = ColPali.from_pretrained(
            "vidore/colpali-v1.3",
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        ).eval()
        
        self.processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        self.collection_name = collection_name
        self._setup_collection()

    def _setup_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
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

    def _extract_page_text(self, page):
        """
        Extract text from a PDF page with enhanced structure preservation
        """
        # Get plain text
        text_plain = page.get_text("text")
        
        # Get text with HTML formatting
        text_html = page.get_text("html")
        
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
        
        return {
            "text_plain": text_plain,
            "text_html": text_html,
            "blocks": structured_blocks,
            "layout": text_dict,
        }

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
            patch_size=self.model.patch_size
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

    def ingest_pdf(self, pdf_path, batch_size=4):
        """Process entire PDF and store vectors"""
        doc = fitz.open(pdf_path)
        
        # Extract document metadata
        metadata = doc.metadata
        
        for page_num in range(len(doc)):
            
            page = doc[page_num]
            
            # Extract text with enhanced structure
            text_data = self._extract_page_text(page)
            
            # Process page as image
            image = self._convert_page_to_image(page)
            vectors = self._process_page(image)
            
            # Store in Qdrant with enhanced metadata
            self.client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(
                    id=page_num,
                    vector=vectors,
                    payload={
                        "pdf_path": pdf_path,
                        "page_number": page_num,
                        "metadata": metadata,
                        "text_content": text_data,
                        "page_dimensions": {
                            "width": page.rect.width,
                            "height": page.rect.height
                        }
                    }
                )]
            )
        
        doc.close()

    def search(self, query, limit=10, prefetch_limit=100):
        """Search through processed PDFs"""
        processed_query = self.processor.process_queries([query]).to(self.model.device)
        query_embedding = self.model(**processed_query)[0]
        query_embedding = query_embedding.to(torch.float32).detach().cpu().numpy()
        
        response = self.client.query_points(
            collection_name=self.collection_name,
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
        return response.points