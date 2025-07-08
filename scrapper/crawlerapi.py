from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId
import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import contextlib
from asyncio import Event

from eaglecrawler import EagleCrawler

app = FastAPI(title="EagleCrawler API")
logger = logging.getLogger("uvicorn")

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "")
if MONGODB_URL == "":
    raise Exception("Mongodb connection string is empty")
DATABASE_NAME = "crawler_jobs"
COLLECTION_NAME = "jobs"

# Thread pool for database operations
executor = ThreadPoolExecutor(max_workers=20)

# MongoDB client (synchronous)
client = MongoClient(MONGODB_URL)
database = client[DATABASE_NAME]
jobs_collection = database[COLLECTION_NAME]

# Global dictionary to track active tasks
active_tasks: Dict[str, asyncio.Task] = {}
task_lock = asyncio.Lock()

# Job status enum
class JobStatus(str, Enum): 
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Request models
class CrawlRequest(BaseModel):
    urls: Union[str,List[str]] = Field(...,description="URL or List of URLs to crawl")
    mode: str = Field("text", description="Web crawling mode, 'visual', 'text', or 'both'")
    output_dir: str = Field("crawler_output", description="Output directory path")
    # A4 visual settings
    page_width: int = Field(1920, description="Width in pixel count for screenshot resolution")
    min_overlap: int = Field(50, description="Minimum allowed page overlap in pixels")
    smart_splitting: bool = Field(True, description="Bool to enable content-aware splitting")
    preserve_context: bool = Field(True, description="Bool to prevent cutting important elements")
    # General settings
    wait_time: int = Field(3000, description="Wait time before capture (ms)")
    headless: bool = Field(True, description="Run browser headlessly")
    max_pages: int = Field(10, description="Maximum pages to crawl")
    page_timeout: int = Field(60000, description="Page load timeout (ms)")
    navigation_timeout: int = Field(30000, description="Navigation timeout (ms)")
    retry_attempts: int = Field(2, description="Number of retry attempts for failures")
    # Text mode specific settings
    extract_links: bool = Field(True, description="Bool to extract hyperlinks from pages")
    extract_images: bool = Field(True, description="Bool to extract image metadata")
    clean_text: bool = Field(True, description="Bool to clean extracted text content")
    save_html: bool = Field(False, description="Bool to save raw HTML content")
    content_selectors: List[str] = Field(None, description="List of CSS selectors for content extraction")
    # Recursive crawling settings
    max_depth: int = Field(1, description="Maximum depth to crawl (1 = no recursion, 2 = one level deep, etc.)")
    same_domain_only: bool = Field(True, description="Bool to only crawl URLs from the same domain as starting URLs")
    url_patterns: List[str] = Field(None, description="List of regex patterns that URLs must match to be crawled")
    exclude_patterns: List[str] = Field(None, description="List of regex patterns to exclude from crawling")
    delay_between_requests: float = Field(1.0, description="Delay in seconds between requests to be respectful")
    # Content detection settings
    min_content_length: int = Field(300, description="Minimum number of characters to consider a block being valid content")
    # Boilerplate removal settings
    boilerplate_shingle_size: int = Field(5, description="Lines per shingle for boilerplate detection")  
    boilerplate_threshold: float = Field(0.5, description="Minimum percentage of pages containing shingle to be considered boilerplate")

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    progress: int
    params: Dict[str, Any]

# Helper functions to run sync MongoDB operations in thread pool
async def run_in_executor(func, *args, **kwargs):
    """Run synchronous function in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

# Database operations (synchronous functions)
def create_job_in_db_sync(params: Dict[str, Any]) -> str:
    """Create job in MongoDB (synchronous)"""
    job_doc = {
        "status": JobStatus.PENDING,
        "params": params,
        "result": None,
        "error": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "completed_at": None,
        "progress": 0
    }
    
    result = jobs_collection.insert_one(job_doc)
    return str(result.inserted_id)

def update_job_status_sync(job_id: str, status: JobStatus, **kwargs):
    """Update job status in MongoDB (synchronous)"""
    update_data = {
        "status": status,
        "updated_at": datetime.now()
    }
    update_data.update(kwargs)
    
    jobs_collection.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": update_data}
    )

def get_job_from_db_sync(job_id: str) -> Optional[Dict]:
    """Get job from MongoDB (synchronous)"""
    if not ObjectId.is_valid(job_id):
        return None
    
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if job:
        job["job_id"] = str(job["_id"])
        job.pop("_id")
    return job

def get_all_jobs_from_db_sync(skip: int = 0, limit: int = 100, status: Optional[JobStatus] = None) -> List[Dict]:
    """Get all jobs from MongoDB (synchronous)"""
    filter_query = {}
    if status:
        filter_query["status"] = status
    
    cursor = jobs_collection.find(filter_query).sort("created_at", DESCENDING).skip(skip).limit(limit)
    jobs = list(cursor)
    
    for job in jobs:
        job["job_id"] = str(job["_id"])
        job.pop("_id")
    
    return jobs

def delete_job_from_db_sync(job_id: str) -> bool:
    """Delete job from MongoDB (synchronous)"""
    if not ObjectId.is_valid(job_id):
        return False
    
    result = jobs_collection.delete_one({"_id": ObjectId(job_id)})
    return result.deleted_count > 0

def get_job_stats_sync() -> Dict:
    """Get job statistics (synchronous)"""
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
    
    return {
        "total_jobs": total_jobs,
        "by_status": stats
    }

# Async wrappers for database operations
async def create_job_in_db(params: Dict[str, Any]) -> str:
    return await run_in_executor(create_job_in_db_sync, params)

async def update_job_status(job_id: str, status: JobStatus, **kwargs):
    return await run_in_executor(update_job_status_sync, job_id, status, **kwargs)

async def get_job_from_db(job_id: str) -> Optional[Dict]:
    return await run_in_executor(get_job_from_db_sync, job_id)

async def get_all_jobs_from_db(skip: int = 0, limit: int = 100, status: Optional[JobStatus] = None) -> List[Dict]:
    return await run_in_executor(get_all_jobs_from_db_sync, skip, limit, status)

async def delete_job_from_db(job_id: str) -> bool:
    return await run_in_executor(delete_job_from_db_sync, job_id)

async def get_job_stats() -> Dict:
    return await run_in_executor(get_job_stats_sync)

# Function to run the crawl in an asyncio task
async def run_crawl(job_id: str, crawl_request: CrawlRequest, cancellation_event: asyncio.Event):
    """Run the crawl operation in an asyncio task"""
    try:
        logger.info(f"Starting crawl job: {job_id}")
        await update_job_status(job_id, JobStatus.RUNNING, progress=0)
        
        # Create crawler instance with all parameters
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
            cancellation_event=cancellation_event  # Pass cancellation event to crawler
        )
        
        # Run the crawl asynchronously
        results = await crawler.crawl(crawl_request.urls)
        
        # Check if job was cancelled during processing
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

# Initialize database indexes
def init_database():
    """Initialize database indexes"""
    try:
        # Create indexes for better performance
        jobs_collection.create_index("status")
        jobs_collection.create_index("created_at")
        jobs_collection.create_index([("status", ASCENDING), ("created_at", DESCENDING)])
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    await run_in_executor(init_database)
    logger.info("Connected to MongoDB with PyMongo")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    # Cancel all active tasks
    async with task_lock:
        for job_id, task in list(active_tasks.items()):
            task.cancel()
            logger.info(f"Cancelling task for job {job_id} during shutdown")
    
    # Give tasks time to handle cancellation
    await asyncio.sleep(1)
    
    client.close()
    executor.shutdown(wait=True)
    logger.info("Disconnected from MongoDB")

@app.post("/jobs", response_model=JobResponse)
async def create_job(crawl_request: CrawlRequest):
    """
    Create a new background job
    """
    try:
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
    
    except Exception as e:  
        logger.error(f"Failed to create job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get job status by ID
    """
    job = await get_job_from_db(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(**job)

@app.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs(
    skip: int = 0, 
    limit: int = 100, 
    status: Optional[JobStatus] = None
):
    """
    List all jobs with optional filtering
    """
    jobs = await get_all_jobs_from_db(skip=skip, limit=limit, status=status)
    return [JobStatusResponse(**job) for job in jobs]

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job record
    """
    # Cancel the task if it's running
    async with task_lock:
        if job_id in active_tasks:
            task, cancellation_event = active_tasks[job_id]
            task.cancel()
            del active_tasks[job_id]
    
    # Delete from database
    success = await delete_job_from_db(job_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"message": "Job deleted successfully"}

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a pending or running job
    """
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
    
    # For jobs that are running but not in active_tasks (shouldn't happen normally)
    await update_job_status(job_id, JobStatus.CANCELLED, error="Job cancelled externally")
    return {"message": "Job marked as cancelled"}

@app.get("/stats")
async def get_job_statistics():
    """
    Get job statistics
    """
    stats = await get_job_stats()       
    return stats

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Test MongoDB connection
        await run_in_executor(lambda: client.admin.command('ping'))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)