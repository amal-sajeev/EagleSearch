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

from eaglecrawler import EagleCrawler

app = FastAPI(title="FastAPI + MongoDB Job System (Pure PyMongo)")

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "")
if MONGODB_URL == "":
    raise Exception("Mongodb connection string is empty")
DATABASE_NAME = "job_system"
COLLECTION_NAME = "jobs"

# Thread pool for database operations
executor = ThreadPoolExecutor(max_workers=20)

# MongoDB client (synchronous)
client = MongoClient(MONGODB_URL)
database = client[DATABASE_NAME]
jobs_collection = database[COLLECTION_NAME]

# Job status enum
class JobStatus(str, Enum): 
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Request models
class CrawlRequest(BaseModel):
    urls: Union[str,List[str]] = Field(...,description="URL or List of URLs to crawl")
    mode: str = Field("text", description = "Web crawling mode, 'visual', 'text', or 'both' ")
    output_dir: str = Field("crawler_output", description = "Output directory path")
    # A4 visual settings
    page_width: int = Field(1920, description = "Width in pixel count for screenshot resolution.")
    min_overlap: int = Field(50, description = "Minimum allowed page overlap in pixels")
    smart_splitting: bool = Field(True, description = "Bool to enable content-aware splitting")
    preserve_context: bool = Field(True, description = "Bool to prevent cutting important elements")
    # General settings
    wait_time: int = Field(3000, description = "Wait time before capture (ms)")
    headless: bool = Field(True, description = "Run browser headlessly")
    max_pages: int = Field(10, description = "Maximum pages to crawl")
    page_timeout: int = Field(60000, description = "Page load timeout (ms)")
    navigation_timeout: int = Field(30000, description = "Navigation timeout (ms)")
    retry_attempts: int = Field(2, description = "Number of retry attempts for failures")
    # Text mode specific settings
    extract_links: bool = Field(True, description = "Bool to extract hyperlinks from pages")
    extract_images: bool = Field(True, description = "Bool to extract image metadata")
    clean_text: bool = Field(True, description = "Bool to clean extracted text content")
    save_html: bool = Field(False, description = "Bool to save raw HTML content")
    content_selectors: List[str] = Field(None, description = "List of CSS selectors for content extraction")
    # Recursive crawling settings
    max_depth: int = Field(1, description = "Maximum depth to crawl (1 = no recursion, 2 = one level deep, etc.)")
    same_domain_only: bool = Field(True, description = "Bool to only crawl URLs from the same domain as starting URLs")
    url_patterns: List[str] = Field(None, description = "List of regex patterns that URLs must match to be crawled")
    exclude_patterns: List[str] = Field(None, description = "List of regex patterns to exclude from crawling")
    delay_between_requests: float = Field(1.0, description = "Delay in seconds between requests to be respectful")
    # Content detection settings
    min_content_length: int = Field(300, description = "Minimum number of characters to consider a block being valid content")
    # Boilerplate removal settings
    boilerplate_shingle_size: int = Field(5, description = "Lines per shingle for boilerplate detection")  
    boilerplate_threshold: float = Field(0.5, description = "Minimum percentage of pages containing shingle to be considered boilerplate")

    param1: str = Field(..., description="First parameter")
    param2: int = Field(..., ge=1, description="Second parameter (must be >= 1)")
    param3: Optional[str] = Field(None, description="Optional third parameter")

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
    return await loop.run_in_executor(executor, func, *args, **kwargs)

# Database operations (synchronous functions)
def create_job_in_db_sync(params: Dict[str, Any]) -> str:
    """Create job in MongoDB (synchronous)"""
    job_doc = {
        "status": JobStatus.PENDING,
        "params": params,
        "result": None,
        "error": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "completed_at": None,
        "progress": 0
    }
    
    result = jobs_collection.insert_one(job_doc)
    return str(result.inserted_id)

def update_job_status_sync(job_id: str, status: JobStatus, **kwargs):
    """Update job status in MongoDB (synchronous)"""
    update_data = {
        "status": status,
        "updated_at": datetime.utcnow()
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

# Long-running function that runs in a separate thread
def long_running_function_sync(job_id: str,
        urls:Union[str,List[str]],
        mode: str = "text",
        output_dir: str = "crawler_output",
        # A4 visual settings
        page_width: int = 1920,
        min_overlap: int = 50,
        smart_splitting: bool = True,
        preserve_context: bool = True,
        # General settings
        wait_time: int = 3000,
        headless: bool = True,
        max_pages: int = 10,
        page_timeout: int = 60000,
        navigation_timeout: int = 30000,
        retry_attempts: int = 2,
        # Text mode specific settings
        extract_links: bool = True,
        extract_images: bool = True,
        clean_text: bool = True,
        save_html: bool = False,
        content_selectors: List[str] = None,
        # Recursive crawling settings
        max_depth: int = 1,
        same_domain_only: bool = True,
        url_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        delay_between_requests: float = 1.0,
        # Content detection settings
        min_content_length: int = 300,  # Minimum characters to consider valid content
        # Boilerplate removal settings
        boilerplate_shingle_size: int = 5,  # Lines per shingle for boilerplate detection
        boilerplate_threshold: float = 0.5  # Min percentage of pages containing shingle to be considered boilerplate
        ):
    """
    Long-running function that runs synchronously in a separate thread
    """
    try:
        print(f"Starting job {job_id}")
        
        # Update job status to running
        update_job_status_sync(job_id, JobStatus.RUNNING, progress=0)
        
        # Begin crawl
        results = EagleCrawler.crawl(urls)
        result = []
        # Simulate final result
        for res in results:
            result.append({
                "url" : res.url,
                "depth" : res.metadata.get('crawl_depth', 0) if res.metadata else 0,
                "status" : '✅ Success' if not res.error else '❌ Error',
                "error": res.error if res.error else "",
                "text_length": len(res.text_content) if not res.error else 0,
                "links_found": len(res.links) if not res.error else 0,
                "boilerplate_removed": bp['removed_lines'] if "boilerplate_removed" in res.metadata else 0,
                "boilerplate_percent": bp['removal_percentage'],
                "screenshot_paths": res.screenshot_paths,
                "crawl_time": res.timestamp,
                "text_content": res.text_content,
                "html_content": res.html_content,
                "status_code": res.status_code,
                "title": res.title,
                "page_count": res.page_count
            })
        
        # Update job with completion
        update_job_status_sync(
            job_id, 
            JobStatus.COMPLETED, 
            result=result,
            completed_at=datetime.utcnow(),
            progress=100
        )
        
        print(f"Job {job_id} completed successfully")
        
    except Exception as e:
        # Handle errors
        update_job_status_sync(
            job_id,
            JobStatus.FAILED,
            error=str(e),
            completed_at=datetime.utcnow()
        )
        print(f"Job {job_id} failed: {str(e)}")

# Async wrapper for the long-running function
async def long_running_function(job_id: str,
        urls:Union[str,List[str]],
        mode: str = "text",
        output_dir: str = "crawler_output",
        # A4 visual settings
        page_width: int = 1920,
        min_overlap: int = 50,
        smart_splitting: bool = True,
        preserve_context: bool = True,
        # General settings
        wait_time: int = 3000,
        headless: bool = True,
        max_pages: int = 10,
        page_timeout: int = 60000,
        navigation_timeout: int = 30000,
        retry_attempts: int = 2,
        # Text mode specific settings
        extract_links: bool = True,
        extract_images: bool = True,
        clean_text: bool = True,
        save_html: bool = False,
        content_selectors: List[str] = None,
        # Recursive crawling settings
        max_depth: int = 1,
        same_domain_only: bool = True,
        url_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        delay_between_requests: float = 1.0,
        # Content detection settings
        min_content_length: int = 300,  # Minimum characters to consider valid content
        # Boilerplate removal settings
        boilerplate_shingle_size: int = 5,  # Lines per shingle for boilerplate detection
        boilerplate_threshold: float = 0.5  # Min percentage of pages containing shingle to be considered boilerplate
        ):
    """
    Async wrapper that runs the long-running function in a separate thread
    """
    await run_in_executor(EagleCrawler.crawl, job_id, urls, mode, output_dir, page_width, min_overlap, smart_splitting, preserve_context, wait_time, headless, max_pages, page_timeout, navigation_timeout, retry_attempts, extract_links, extract_images, clean_text, save_html, content_selectors, max_depth, same_domain_only, url_patterns, exclude_patterns, delay_between_requests, min_content_length, boilerplate_shingle_size, boilerplate_threshold)

# Initialize database indexes
def init_database():
    """Initialize database indexes"""
    try:
        # Create indexes for better performance
        jobs_collection.create_index("status")
        jobs_collection.create_index("created_at")
        jobs_collection.create_index([("status", ASCENDING), ("created_at", DESCENDING)])
        print("Database indexes created successfully")
    except Exception as e:
        print(f"Error creating indexes: {e}")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    await run_in_executor(init_database)
    print("Connected to MongoDB with PyMongo")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    client.close()
    executor.shutdown(wait=True)
    print("Disconnected from MongoDB")

@app.post("/jobs", response_model=JobResponse)
async def create_job(crawl_request: CrawlRequest, background_tasks: BackgroundTasks):
    """
    Create a new background job
    """
    try:
        # Create job in database
        job_id = await create_job_in_db(crawl_request.dict())
        
        # Add the long-running function to background tasks
        background_tasks.add_task(
            long_running_function,
            job_id,
            crawl_request.param1,
            crawl_request.param2,
            crawl_request.param3
        )
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Job created successfully"
        )
    
    except Exception as e:  
        raise HTTPException(status_code=500, detail=f"Failed to create  ob: {str(e)}")

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
    success = await delete_job_from_db(job_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"message": "Job deleted successfully"}

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a pending job
    """
    job = await get_job_from_db(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
    
    await update_job_status(job_id, JobStatus.FAILED, error="Job cancelled by user")
    
    return {"message": "Job cancelled successfully"}

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
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

# Direct synchronous endpoint example (if you want to avoid async entirely)
@app.get("/jobs/{job_id}/sync")
def get_job_status_sync(job_id: str):
    """
    Synchronous version of get job status (no async/await)
    """
    job = get_job_from_db_sync(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(**job)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)