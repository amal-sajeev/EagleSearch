"""
EagleCrawler API v2.0

This FastAPI application provides a robust web crawling service with:
- Job management (creation, monitoring, cancellation)
- Real-time progress streaming via SSE
- Connection pool monitoring and health checks
- MongoDB persistence for job tracking
- Thread-safe task management
- Comprehensive input validation

Key Components:
1. FastAPI application with optimized MongoDB connection pooling
2. Job lifecycle management (pending, running, completed, failed, cancelled)
3. StreamingResponse for real-time job updates
4. Connection pool health monitoring
5. Background tasks for periodic health checks
6. Input validation with Pydantic models
7. Asynchronous task execution with cancellation support
8. Database indexing and optimization
9. System resource monitoring
10. Stale job cleanup mechanism

Environment Variables:
- MONGODB_URL: MongoDB connection string
- MONGO_MAX_POOL_SIZE: Maximum connection pool size (default: 50)
- MONGO_MIN_POOL_SIZE: Minimum connection pool size (default: 10)
- DB_THREAD_POOL_SIZE: Thread pool size for DB operations (default: 20)
- ALLOW_LOCAL_URLS: Enable crawling of local URLs (default: false)
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator, AnyHttpUrl
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import ObjectId
import asyncio
import time
import json
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from enum import Enum
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import contextlib
from asyncio import Event
import re
import validators
from urllib.parse import urlparse
import psutil

from eaglecrawler import EagleCrawler

# Initialize FastAPI application
app = FastAPI(title="EagleCrawler API", version="2.0.0")
logger = logging.getLogger("uvicorn")

# MongoDB connection with optimized pool settings
MONGODB_URL = os.getenv("MONGODB_URL", "")
if MONGODB_URL == "":
    raise Exception("Mongodb connection string is empty")

DATABASE_NAME = "crawler_jobs"
COLLECTION_NAME = "jobs"

# Enhanced MongoDB client with connection pool optimization
client = MongoClient(
    MONGODB_URL,
    maxPoolSize=int(os.getenv("MONGO_MAX_POOL_SIZE", "50")),
    minPoolSize=int(os.getenv("MONGO_MIN_POOL_SIZE", "10")),
    maxIdleTimeMS=int(os.getenv("MONGO_MAX_IDLE_TIME_MS", "45000")),
    connectTimeoutMS=int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "10000")),
    serverSelectionTimeoutMS=int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "10000")),
    socketTimeoutMS=int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "20000")),
    retryWrites=True,
    retryReads=True,
    w="majority",
    readPreference="primary"
)

database = client[DATABASE_NAME]
jobs_collection = database[COLLECTION_NAME]

# Thread pool for database operations
executor = ThreadPoolExecutor(max_workers=int(os.getenv("DB_THREAD_POOL_SIZE", "20")))

# Global dictionary to track active tasks and streaming clients
active_tasks: Dict[str, asyncio.Task] = {}
streaming_clients: Dict[str, List[asyncio.Queue]] = {}
task_lock = asyncio.Lock()

# Connection pool monitoring
class ConnectionPoolMonitor:
    """Monitors MongoDB connection pool health and statistics"""
    def __init__(self):
        self.start_time = datetime.now()
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "last_health_check": None,
            "health_status": "unknown"
        }
    
    def get_pool_stats(self) -> Dict:
        """Get current connection pool statistics"""
        try:
            server_info = client.server_info()
            pool_stats = {
                "pool_size": client.max_pool_size,
                "min_pool_size": client.min_pool_size,
                "max_idle_time": client.max_idle_time_ms,
                "connect_timeout": client.connect_timeout_ms,
                "server_selection_timeout": client.server_selection_timeout_ms,
                "current_connections": len(client.nodes),
                "server_info": {
                    "version": server_info.get("version"),
                    "uptime": server_info.get("uptime")
                },
                "last_updated": datetime.now().isoformat()
            }
            return pool_stats
        except Exception as e:
            logger.error(f"Failed to get pool stats: {e}")
            return {"error": str(e)}
    
    async def check_connection_health(self) -> bool:
        """Check MongoDB connection health"""
        try:
            await run_in_executor(lambda: client.admin.command('ping'))
            self.connection_stats["health_status"] = "healthy"
            self.connection_stats["last_health_check"] = datetime.now().isoformat()
            return True
        except Exception as e:
            self.connection_stats["health_status"] = "unhealthy"
            self.connection_stats["last_health_check"] = datetime.now().isoformat()
            logger.error(f"MongoDB health check failed: {e}")
            return False

pool_monitor = ConnectionPoolMonitor()

# Job status enumeration
class JobStatus(str, Enum): 
    """Represents possible states of a crawl job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Enhanced input validation models
class CrawlRequest(BaseModel):
    """
    Validated crawl request parameters
    
    Attributes:
    urls: Single URL or list of URLs to crawl
    mode: Crawling mode ('text', 'visual', or 'both')
    output_dir: Directory to store crawl results
    page_width: Browser viewport width (800-3840px)
    min_overlap: Minimum screenshot overlap (0-500px)
    smart_splitting: Enable content-aware page splitting
    preserve_context: Prevent cutting important elements
    wait_time: Page load wait time (100-30000ms)
    headless: Run browser in headless mode
    max_pages: Maximum pages to crawl (1-1000)
    page_timeout: Page load timeout (5000-300000ms)
    navigation_timeout: Navigation timeout (1000-120000ms)
    retry_attempts: Failed request retry count (0-10)
    extract_links: Extract hyperlinks from pages
    extract_images: Extract image metadata
    clean_text: Clean extracted text content
    save_html: Save raw HTML content
    content_selectors: CSS selectors for content extraction
    max_depth: Maximum crawl depth (1-10)
    same_domain_only: Restrict to same-domain URLs
    url_patterns: Regex patterns for URL inclusion
    exclude_patterns: Regex patterns for URL exclusion
    delay_between_requests: Delay between requests (0.1-10.0s)
    min_content_length: Minimum content length (50-10000 chars)
    boilerplate_shingle_size: Lines per shingle (1-20)
    boilerplate_threshold: Boilerplate removal threshold (0.1-1.0)
    """
    urls: Union[str, List[str]] = Field(..., description="URL or List of URLs to crawl")
    mode: str = Field("text", description="Web crawling mode: 'visual', 'text', or 'both'")
    output_dir: str = Field("crawler_output", description="Output directory path")
    
    # A4 visual settings with validation
    page_width: int = Field(1920, ge=800, le=3840, description="Width in pixels (800-3840)")
    min_overlap: int = Field(50, ge=0, le=500, description="Minimum overlap in pixels (0-500)")
    smart_splitting: bool = Field(True, description="Enable content-aware splitting")
    preserve_context: bool = Field(True, description="Prevent cutting important elements")
    
    # General settings with validation
    wait_time: int = Field(3000, ge=100, le=30000, description="Wait time before capture (100-30000ms)")
    headless: bool = Field(True, description="Run browser headlessly")
    max_pages: int = Field(10, ge=1, le=1000, description="Maximum pages to crawl (1-1000)")
    page_timeout: int = Field(60000, ge=5000, le=300000, description="Page load timeout (5000-300000ms)")
    navigation_timeout: int = Field(30000, ge=1000, le=120000, description="Navigation timeout (1000-120000ms)")
    retry_attempts: int = Field(2, ge=0, le=10, description="Number of retry attempts (0-10)")
    
    # Text mode specific settings
    extract_links: bool = Field(True, description="Extract hyperlinks from pages")
    extract_images: bool = Field(True, description="Extract image metadata")
    clean_text: bool = Field(True, description="Clean extracted text content")
    save_html: bool = Field(False, description="Save raw HTML content")
    content_selectors: Optional[List[str]] = Field(None, description="CSS selectors for content extraction")
    
    # Recursive crawling settings with validation
    max_depth: int = Field(1, ge=1, le=10, description="Maximum crawl depth (1-10)")
    same_domain_only: bool = Field(True, description="Only crawl URLs from the same domain")
    url_patterns: Optional[List[str]] = Field(None, description="Regex patterns for URL matching")
    exclude_patterns: Optional[List[str]] = Field(None, description="Regex patterns to exclude")
    delay_between_requests: float = Field(1.0, ge=0.1, le=10.0, description="Delay between requests (0.1-10.0s)")
    
    # Content detection settings
    min_content_length: int = Field(300, ge=50, le=10000, description="Minimum content length (50-10000)")
    
    # Boilerplate removal settings
    boilerplate_shingle_size: int = Field(5, ge=1, le=20, description="Lines per shingle (1-20)")
    boilerplate_threshold: float = Field(0.5, ge=0.1, le=1.0, description="Boilerplate threshold (0.1-1.0)")
    
    @validator('urls')
    def validate_urls(cls, v):
        """Validate URLs format and accessibility"""
        if isinstance(v, str):
            v = [v]
        
        if not v or len(v) == 0:
            raise ValueError("At least one URL is required")
        
        if len(v) > 100:  # Limit number of URLs
            raise ValueError("Maximum 100 URLs allowed")
        
        validated_urls = []
        for url in v:
            if not isinstance(url, str):
                raise ValueError("URLs must be strings")
            
            # Basic URL validation
            if not validators.url(url):
                raise ValueError(f"Invalid URL format: {url}")
            
            # Check protocol
            parsed = urlparse(url)
            if parsed.scheme not in ['http', 'https']:
                raise ValueError(f"Only HTTP and HTTPS protocols allowed: {url}")
            
            # Check for suspicious domains (basic security)
            if any(blocked in parsed.netloc.lower() for blocked in ['localhost', '127.0.0.1', '0.0.0.0']):
                if not os.getenv("ALLOW_LOCAL_URLS", "false").lower() == "true":
                    raise ValueError(f"Local URLs not allowed: {url}")
            
            validated_urls.append(url)
        
        return validated_urls
    
    @validator('mode')
    def validate_mode(cls, v):
        """Validate crawling mode"""
        if v not in ['text', 'visual', 'both']:
            raise ValueError("Mode must be 'text', 'visual', or 'both'")
        return v
    
    @validator('output_dir')
    def validate_output_dir(cls, v):
        """Validate output directory"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Output directory cannot be empty")
        
        # Sanitize path
        sanitized = re.sub(r'[<>:"|?*]', '_', v)
        if sanitized != v:
            raise ValueError("Output directory contains invalid characters")
        
        return sanitized
    
    @validator('content_selectors')
    def validate_content_selectors(cls, v):
        """Validate CSS selectors"""
        if v is None:
            return v
        
        if len(v) > 20:
            raise ValueError("Maximum 20 content selectors allowed")
        
        for selector in v:
            if not isinstance(selector, str) or len(selector.strip()) == 0:
                raise ValueError("Content selectors must be non-empty strings")
        
        return v
    
    @validator('url_patterns', 'exclude_patterns')
    def validate_regex_patterns(cls, v):
        """Validate regex patterns"""
        if v is None:
            return v
        
        if len(v) > 10:
            raise ValueError("Maximum 10 regex patterns allowed")
        
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        
        return v

class JobResponse(BaseModel):
    """Response model for job creation"""
    job_id: str
    status: JobStatus
    message: str

class JobStatusResponse(BaseModel):
    """Detailed job status response"""
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    progress: int
    params: Dict[str, Any]

class JobUpdateMessage(BaseModel):
    """Real-time job update message for SSE"""
    job_id: str
    status: JobStatus
    progress: int
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None

class ConnectionPoolStats(BaseModel):
    """MongoDB connection pool statistics"""
    pool_size: int
    min_pool_size: int
    max_idle_time: int
    connect_timeout: int
    server_selection_timeout: int
    current_connections: int
    health_status: str
    last_health_check: Optional[str]
    uptime_seconds: int

# Helper functions to run sync MongoDB operations in thread pool
async def run_in_executor(func, *args, **kwargs):
    """Run synchronous function in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

# Enhanced database operations with better error handling
def create_job_in_db_sync(params: Dict[str, Any]) -> str:
    """Create job in MongoDB (synchronous)"""
    try:
        job_doc = {
            "status": JobStatus.PENDING,
            "params": params,
            "result": None,
            "error": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "completed_at": None,
            "progress": 0,
            "last_heartbeat": datetime.now()
        }
        
        result = jobs_collection.insert_one(job_doc)
        pool_monitor.connection_stats["total_connections"] += 1
        return str(result.inserted_id)
    except Exception as e:
        pool_monitor.connection_stats["failed_connections"] += 1
        raise e

def update_job_status_sync(job_id: str, status: JobStatus, **kwargs):
    """Update job status in MongoDB (synchronous)"""
    try:
        update_data = {
            "status": status,
            "updated_at": datetime.now(),
            "last_heartbeat": datetime.now()
        }
        update_data.update(kwargs)
        
        result = jobs_collection.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise ValueError(f"Job {job_id} not found")
        
        return result.modified_count > 0
    except Exception as e:
        pool_monitor.connection_stats["failed_connections"] += 1
        raise e

def get_job_from_db_sync(job_id: str) -> Optional[Dict]:
    """Get job from MongoDB (synchronous)"""
    try:
        if not ObjectId.is_valid(job_id):
            return None
        
        job = jobs_collection.find_one({"_id": ObjectId(job_id)})
        if job:
            job["job_id"] = str(job["_id"])
            job.pop("_id")
        return job
    except Exception as e:
        pool_monitor.connection_stats["failed_connections"] += 1
        raise e

def get_all_jobs_from_db_sync(skip: int = 0, limit: int = 100, status: Optional[JobStatus] = None) -> List[Dict]:
    """Get all jobs from MongoDB with pagination (synchronous)"""
    try:
        filter_query = {}
        if status:
            filter_query["status"] = status
        
        cursor = jobs_collection.find(
            filter_query,
            projection={"params.urls": 0}  # Exclude large URL lists from list view
        ).sort("created_at", DESCENDING).skip(skip).limit(limit)
        
        jobs = list(cursor)
        
        for job in jobs:
            job["job_id"] = str(job["_id"])
            job.pop("_id")
        
        return jobs
    except Exception as e:
        pool_monitor.connection_stats["failed_connections"] += 1
        raise e

def delete_job_from_db_sync(job_id: str) -> bool:
    """Delete job from MongoDB (synchronous)"""
    try:
        if not ObjectId.is_valid(job_id):
            return False
        
        result = jobs_collection.delete_one({"_id": ObjectId(job_id)})
        return result.deleted_count > 0
    except Exception as e:
        pool_monitor.connection_stats["failed_connections"] += 1
        raise e

def get_job_stats_sync() -> Dict:
    """Get job statistics (synchronous)"""
    try:
        pipeline = [
            {"$group": {
                "_id": "$status",
                "count": {"$sum": 1}
            }}
        ]
        
        stats = {}
        for doc in jobs_collection.aggregate(pipeline):
            stats[doc["_id"]] = doc["count"]
        
        total_jobs = jobs_collection.count_documents({})
        
        # Add system stats
        system_stats = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        return {
            "total_jobs": total_jobs,
            "by_status": stats,
            "system": system_stats,
            "database": pool_monitor.connection_stats
        }
    except Exception as e:
        pool_monitor.connection_stats["failed_connections"] += 1
        raise e

# Async wrappers for database operations
async def create_job_in_db(params: Dict[str, Any]) -> str:
    """Async wrapper for job creation"""
    return await run_in_executor(create_job_in_db_sync, params)

async def update_job_status(job_id: str, status: JobStatus, **kwargs):
    """Async wrapper for job status update with broadcast"""
    result = await run_in_executor(update_job_status_sync, job_id, status, **kwargs)
    
    # Broadcast update to streaming clients
    await broadcast_job_update(job_id, status, kwargs.get('progress', 0), f"Job status updated to {status}")
    
    return result

async def get_job_from_db(job_id: str) -> Optional[Dict]:
    """Async wrapper for job retrieval"""
    return await run_in_executor(get_job_from_db_sync, job_id)

async def get_all_jobs_from_db(skip: int = 0, limit: int = 100, status: Optional[JobStatus] = None) -> List[Dict]:
    """Async wrapper for job listing"""
    return await run_in_executor(get_all_jobs_from_db_sync, skip, limit, status)

async def delete_job_from_db(job_id: str) -> bool:
    """Async wrapper for job deletion"""
    return await run_in_executor(delete_job_from_db_sync, job_id)

async def get_job_stats() -> Dict:
    """Async wrapper for job statistics"""
    return await run_in_executor(get_job_stats_sync)

# Streaming job updates functionality
async def broadcast_job_update(job_id: str, status: JobStatus, progress: int, message: str, data: Optional[Dict] = None):
    """Broadcast job update to all streaming clients"""
    if job_id not in streaming_clients:
        return
    
    update_message = JobUpdateMessage(
        job_id=job_id,
        status=status,
        progress=progress,
        message=message,
        timestamp=datetime.now(),
        data=data
    )
    
    # Send to all clients streaming this job
    clients_to_remove = []
    for i, client_queue in enumerate(streaming_clients[job_id]):
        try:
            await client_queue.put(update_message)
        except Exception as e:
            logger.error(f"Failed to send update to client {i}: {e}")
            clients_to_remove.append(i)
    
    # Remove failed clients
    for i in reversed(clients_to_remove):
        streaming_clients[job_id].pop(i)
    
    # Clean up if no clients left
    if not streaming_clients[job_id]:
        del streaming_clients[job_id]

async def stream_job_updates(job_id: str) -> AsyncGenerator[str, None]:
    """Stream job updates to client via Server-Sent Events"""
    # Create queue for this client
    client_queue = asyncio.Queue()
    
    # Add client to streaming list
    if job_id not in streaming_clients:
        streaming_clients[job_id] = []
    streaming_clients[job_id].append(client_queue)
    
    try:
        # Send initial job status
        job = await get_job_from_db(job_id)
        if job:
            initial_message = JobUpdateMessage(
                job_id=job_id,
                status=JobStatus(job["status"]),
                progress=job["progress"],
                message=f"Connected to job {job_id}",
                timestamp=datetime.now()
            )
            yield f"data: {initial_message.json()}\n\n"
        
        # Stream updates
        while True:
            try:
                # Wait for update with timeout
                update = await asyncio.wait_for(client_queue.get(), timeout=30.0)
                yield f"data: {update.json()}\n\n"
                
                # Stop streaming if job is completed
                if update.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    break
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                heartbeat = JobUpdateMessage(
                    job_id=job_id,
                    status=JobStatus.RUNNING,
                    progress=0,
                    message="heartbeat",
                    timestamp=datetime.now()
                )
                yield f"data: {heartbeat.json()}\n\n"
                
    except Exception as e:
        error_message = JobUpdateMessage(
            job_id=job_id,
            status=JobStatus.FAILED,
            progress=0,
            message=f"Streaming error: {str(e)}",
            timestamp=datetime.now()
        )
        yield f"data: {error_message.json()}\n\n"
    
    finally:
        # Remove client from streaming list
        if job_id in streaming_clients:
            try:
                streaming_clients[job_id].remove(client_queue)
                if not streaming_clients[job_id]:
                    del streaming_clients[job_id]
            except ValueError:
                pass

# Enhanced crawl function with progress updates
async def run_crawl(job_id: str, crawl_request: CrawlRequest, cancellation_event: asyncio.Event):
    """Run the crawl operation with progress updates"""
    try:
        logger.info(f"Starting crawl job: {job_id}")
        await update_job_status(job_id, JobStatus.RUNNING, progress=0)
        
        # Create crawler instance with validated parameters
        crawler = EagleCrawler(
            mode=crawl_request.mode,
            output_dir=crawl_request.output_dir,
            page_width=crawl_request.page_width,
            min_overlap=crawl_request.min_overlap,
            smart_splitting=crawl_request.smart_splitting,
            preserve_context=crawl_request.preserve_context,
            wait_time=crawl_request.wait_time,
            headless=crawl_request.headless,
            max_pages=crawl_request.max_pages,
            page_timeout=crawl_request.page_timeout,
            navigation_timeout=crawl_request.navigation_timeout,
            retry_attempts=crawl_request.retry_attempts,
            extract_links=crawl_request.extract_links,
            extract_images=crawl_request.extract_images,
            clean_text=crawl_request.clean_text,
            save_html=crawl_request.save_html,
            content_selectors=crawl_request.content_selectors,
            max_depth=crawl_request.max_depth,
            same_domain_only=crawl_request.same_domain_only,
            url_patterns=crawl_request.url_patterns,
            exclude_patterns=crawl_request.exclude_patterns,
            delay_between_requests=crawl_request.delay_between_requests,
            min_content_length=crawl_request.min_content_length,
            boilerplate_shingle_size=crawl_request.boilerplate_shingle_size,
            boilerplate_threshold=crawl_request.boilerplate_threshold,
            cancellation_event=cancellation_event
        )
        
        # Progress tracking
        total_urls = len(crawl_request.urls) if isinstance(crawl_request.urls, list) else 1
        processed_urls = 0
        
        # Run the crawl with progress updates
        results = []
        async for result in crawler.crawl_async(crawl_request.urls):
            if cancellation_event.is_set():
                break
            
            processed_urls += 1
            progress = int((processed_urls / total_urls) * 100)
            
            # Update progress
            await update_job_status(
                job_id, 
                JobStatus.RUNNING, 
                progress=progress,
                data={"processed_urls": processed_urls, "total_urls": total_urls}
            )
            
            results.append(result)
        
        # Check if job was cancelled
        if cancellation_event.is_set():
            logger.info(f"Job {job_id} was cancelled during processing")
            await update_job_status(
                job_id,
                JobStatus.CANCELLED,
                error="Job cancelled during processing",
                completed_at=datetime.now()
            )
            return
        
        # Process results
        formatted_results = []
        for res in results:
            bp = res.metadata.get("boilerplate_removed", {}) if res.metadata else {}
            result_item = {
                "url": res.url,
                "depth": res.metadata.get('crawl_depth', 0) if res.metadata else 0,
                "status": '✅ Success' if not res.error else '❌ Error',
                "error": res.error if res.error else "",
                "text_length": len(res.text_content) if not res.error else 0,
                "links_found": len(res.links) if not res.error else 0,
                "boilerplate_removed": bp.get('removed_lines', 0),
                "boilerplate_percent": bp.get('removal_percentage', 0.0),
                "screenshot_paths": res.screenshot_paths,
                "crawl_time": res.timestamp,
                "text_content": res.text_content,
                "html_content": res.html_content,
                "status_code": res.status_code,
                "title": res.title,
                "page_count": res.page_count
            }
            formatted_results.append(result_item)
        
        # Update job with results
        await update_job_status(
            job_id, 
            JobStatus.COMPLETED, 
            result=formatted_results,
            completed_at=datetime.now(),
            progress=100
        )
        logger.info(f"Job {job_id} completed successfully")
        
    except asyncio.CancelledError:
        logger.info(f"Job {job_id} was cancelled")
        await update_job_status(
            job_id,
            JobStatus.CANCELLED,
            error="Job cancelled by user",
            completed_at=datetime.now()
        )
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        await update_job_status(
            job_id,
            JobStatus.FAILED,
            error=str(e),
            completed_at=datetime.now()
        )
    finally:
        # Clean up task tracking
        async with task_lock:
            if job_id in active_tasks:
                del active_tasks[job_id]

# Enhanced database initialization with optimized indexes
def init_database():
    """Initialize database with optimized indexes"""
    try:
        # Drop existing indexes (except _id)
        indexes_to_drop = []
        for index_name in jobs_collection.index_information():
            if index_name != "_id_":
                indexes_to_drop.append(index_name)
        
        for index_name in indexes_to_drop:
            jobs_collection.drop_index(index_name)
        
        # Create optimized compound indexes
        jobs_collection.create_index([
            ("status", ASCENDING),
            ("created_at", DESCENDING)
        ], name="status_created_compound")
        
        jobs_collection.create_index([
            ("status", ASCENDING),
            ("updated_at", DESCENDING)
        ], name="status_updated_compound")
        
        # Single field indexes
        jobs_collection.create_index("created_at", name="created_at_index")
        jobs_collection.create_index("updated_at", name="updated_at_index")
        jobs_collection.create_index("completed_at", name="completed_at_index")
        jobs_collection.create_index("last_heartbeat", name="heartbeat_index")
        
        # Text search index for error messages
        jobs_collection.create_index([("error", "text")], name="error_text_search")
        
        # TTL index for automatic cleanup of old completed jobs (30 days)
        jobs_collection.create_index(
            "completed_at", 
            expireAfterSeconds=30 * 24 * 60 * 60,  # 30 days
            name="completed_jobs_ttl"
        )
        
        logger.info("Enhanced database indexes created successfully")
        
        # Log index information
        indexes = jobs_collection.index_information()
        logger.info(f"Active indexes: {list(indexes.keys())}")
        
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")

# Health check with connection monitoring
async def check_system_health() -> Dict:
    """Comprehensive system health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # MongoDB health
    try:
        mongo_healthy = await pool_monitor.check_connection_health()
        health_status["checks"]["mongodb"] = {
            "status": "healthy" if mongo_healthy else "unhealthy",
            "pool_stats": pool_monitor.get_pool_stats()
        }
    except Exception as e:
        health_status["checks"]["mongodb"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # System resources
    try:
        health_status["checks"]["system"] = {
            "status": "healthy",
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "active_jobs": len(active_tasks),
            "streaming_clients": sum(len(clients) for clients in streaming_clients.values())
        }
    except Exception as e:
        health_status["checks"]["system"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    return health_status

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    await run_in_executor(init_database)
    await pool_monitor.check_connection_health()
    logger.info("Enhanced EagleCrawler API started with connection pool monitoring")

@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown"""
    # Cancel all active tasks
    async with task_lock:
        for job_id, task in list(active_tasks.items()):
            task.cancel()
            logger.info(f"Cancelling task for job {job_id} during shutdown")
    
    # Close all streaming connections
    for job_id, clients in streaming_clients.items():
        for client_queue in clients:
            try:
                await client_queue.put(JobUpdateMessage(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    progress=0,
                    message="Server shutting down",
                    timestamp=datetime.now()
                ))
            except Exception:
                pass
    
    # Give tasks time to handle cancellation
    await asyncio.sleep(1)
    
    client.close()
    executor.shutdown(wait=True)
    logger.info("Enhanced EagleCrawler API shutdown complete")

@app.post("/jobs", response_model=JobResponse)
async def create_job(crawl_request: CrawlRequest):
    """Create a new crawl job with enhanced validation"""
    try:
        # Additional runtime validation
        if not await pool_monitor.check_connection_health():
            raise HTTPException(status_code=503, detail="Database connection unavailable")
        
        # Create job in database
        job_id = await create_job_in_db(crawl_request.model_dump())
        
        # Create cancellation event for this job
        cancellation_event = asyncio.Event()
        
        # Create and track the crawl task
        task = asyncio.create_task(run_crawl(job_id, crawl_request, cancellation_event))
        
        async with task_lock:
            active_tasks[job_id] = (task, cancellation_event)
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Job created successfully"
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:  
        logger.error(f"Failed to create job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status by ID"""
    try:
        job = await get_job_from_db(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(**job)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/jobs/{job_id}/stream")
async def stream_job_status(job_id: str):
    """Stream real-time job updates via Server-Sent Events"""
    try:
        # Verify job exists
        job = await get_job_from_db(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return StreamingResponse(
            stream_job_updates(job_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start streaming for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start streaming: {str(e)}")

@app.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs(
    skip: int = Query(0, ge=0, description="Number of jobs to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of jobs to return"),
    status: Optional[JobStatus] = Query(None, description="Filter by job status")
):
    """List jobs with pagination and filtering"""
    try:
        jobs = await get_all_jobs_from_db(skip=skip, limit=limit, status=status)
        return [JobStatusResponse(**job) for job in jobs]
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job record"""
    try:
        # Cancel the task if it's running
        async with task_lock:
            if job_id in active_tasks:
                task, cancellation_event = active_tasks[job_id]
                cancellation_event.set()
                task.cancel()
                del active_tasks[job_id]
        
        # Delete from database
        success = await delete_job_from_db(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {"message": "Job deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a pending or running job"""
    try:
        async with task_lock:
            if job_id in active_tasks:
                task, cancellation_event = active_tasks[job_id]
                
                # Set cancellation event to notify crawler
                cancellation_event.set()
                
                # Cancel the asyncio task
                task.cancel()
                
                # Remove from active tasks
                del active_tasks[job_id]
                
                logger.info(f"Job {job_id} cancellation requested")
                return {"message": "Job cancellation requested"}
        
        # If no active task, check job status
        job = await get_job_from_db(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job["status"] == JobStatus.PENDING:
            await update_job_status(job_id, JobStatus.CANCELLED, error="Job cancelled before starting")
            return {"message": "Job was pending and has been cancelled"}
        
        if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed, failed or cancelled job")
        
        # For jobs that are running but not in active_tasks
        await update_job_status(job_id, JobStatus.CANCELLED, error="Job cancelled externally")
        return {"message": "Job marked as cancelled"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@app.get("/stats")
async def get_job_statistics():
    """Get comprehensive job and system statistics"""
    try:
        stats = await get_job_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = await check_system_health()
        
        # Return appropriate HTTP status
        if health_status["status"] == "healthy":
            return health_status
        else:
            raise HTTPException(status_code=503, detail=health_status)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail={"status": "unhealthy", "error": str(e)})

@app.get("/pool-stats")
async def get_connection_pool_stats():
    """Get detailed MongoDB connection pool statistics"""
    try:
        pool_stats = pool_monitor.get_pool_stats()
        health_status = await pool_monitor.check_connection_health()
        
        return {
            "pool_statistics": pool_stats,
            "health_status": "healthy" if health_status else "unhealthy",
            "connection_stats": pool_monitor.connection_stats,
            "active_connections": len(client.nodes),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get pool stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get pool stats: {str(e)}")

@app.get("/monitoring/active-jobs")
async def get_active_jobs():
    """Get information about currently active jobs"""
    try:
        active_job_info = []
        
        async with task_lock:
            for job_id, (task, cancellation_event) in active_tasks.items():
                job_info = {
                    "job_id": job_id,
                    "task_done": task.done(),
                    "task_cancelled": task.cancelled(),
                    "cancellation_requested": cancellation_event.is_set(),
                    "streaming_clients": len(streaming_clients.get(job_id, []))
                }
                
                # Get job details from database
                job = await get_job_from_db(job_id)
                if job:
                    job_info.update({
                        "status": job["status"],
                        "progress": job["progress"],
                        "created_at": job["created_at"].isoformat(),
                        "updated_at": job["updated_at"].isoformat()
                    })
                
                active_job_info.append(job_info)
        
        return {
            "active_jobs_count": len(active_job_info),
            "total_streaming_clients": sum(len(clients) for clients in streaming_clients.values()),
            "active_jobs": active_job_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get active jobs info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active jobs info: {str(e)}")

@app.post("/admin/cleanup-stale-jobs")
async def cleanup_stale_jobs():
    """Clean up stale jobs that may have been interrupted"""
    try:
        # Find jobs that are marked as running but haven't been updated recently
        cutoff_time = datetime.now() - timedelta(hours=2)
        
        def cleanup_stale_sync():
            result = jobs_collection.update_many(
                {
                    "status": JobStatus.RUNNING,
                    "last_heartbeat": {"$lt": cutoff_time}
                },
                {
                    "$set": {
                        "status": JobStatus.FAILED,
                        "error": "Job appears to be stale - marked as failed",
                        "updated_at": datetime.now(),
                        "completed_at": datetime.now()
                    }
                }
            )
            return result.modified_count
        
        cleaned_count = await run_in_executor(cleanup_stale_sync)
        
        return {
            "message": f"Cleaned up {cleaned_count} stale jobs",
            "cleaned_count": cleaned_count,
            "cutoff_time": cutoff_time.isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to cleanup stale jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup stale jobs: {str(e)}")

# Background task for periodic health monitoring
async def periodic_health_check():
    """Periodic health check task"""
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            await pool_monitor.check_connection_health()
        except Exception as e:
            logger.error(f"Periodic health check failed: {e}")

# Start background health monitoring
@app.on_event("startup")
async def start_background_tasks():
    """Start background monitoring tasks"""
    asyncio.create_task(periodic_health_check())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s               - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )