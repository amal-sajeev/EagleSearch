import fitz  # PyMuPDF for PDF processing
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from qdrant_client import QdrantClient, models
from PIL import Image
import numpy as np
import json
import base64
from io import BytesIO
import uuid
import tqdm
import torch
from sentence_transformers import SentenceTransformer
import io
import xml.etree.ElementTree as ET
import zipfile
from typing import Union, List, Dict, Optional, Tuple
import hashlib
import nltk
from sklearn.metrics.pairwise import cosine_similarity

