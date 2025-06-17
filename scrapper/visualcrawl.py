"""
Website Visual Crawler - A Playwright-based module for capturing website screenshots
"""

import asyncio
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass

from playwright.async_api import async_playwright, Page, Browser, BrowserContext


@dataclass
class CrawlResult:
    """Data class to store crawl results"""
    url: str
    screenshot_paths: List[str]  # Changed to list for multiple screenshots
    status_code: int
    title: str
    timestamp: float
    total_height: int = 0
    page_count: int = 0
    error: Optional[str] = None


class WebsiteVisualCrawler:
    """
    A visual website crawler that captures screenshots instead of text data
    """
    
    def __init__(
        self,
        output_dir: str = "screenshots",
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        wait_time: int = 2000,
        headless: bool = True,
        max_pages: int = 10,
        page_timeout: int = 60000,
        navigation_timeout: int = 30000,
        retry_attempts: int = 2,
        split_screenshots: bool = True,
        max_screenshot_width: int = 1920,
        max_screenshot_height: int = 1080,
        overlap_pixels: int = 100
    ):
        """
        Initialize the visual crawler
        
        Args:
            output_dir: Directory to save screenshots
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height  
            wait_time: Time to wait before screenshot (ms)
            headless: Run browser in headless mode
            max_pages: Maximum number of pages to crawl
            page_timeout: Overall page timeout (ms)
            navigation_timeout: Navigation timeout (ms)
            retry_attempts: Number of retry attempts for failed pages
            split_screenshots: Whether to split long pages into chunks
            max_screenshot_width: Maximum width for each screenshot chunk
            max_screenshot_height: Maximum height for each screenshot chunk
            overlap_pixels: Pixels to overlap between chunks to avoid content cutoff
        """
        self.output_dir = Path(output_dir)
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.wait_time = wait_time
        self.headless = headless
        self.max_pages = max_pages
        self.page_timeout = page_timeout
        self.navigation_timeout = navigation_timeout
        self.retry_attempts = retry_attempts
        self.split_screenshots = split_screenshots
        self.max_screenshot_width = max_screenshot_width
        self.max_screenshot_height = max_screenshot_height
        self.overlap_pixels = overlap_pixels
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Track visited URLs to avoid duplicates
        self.visited_urls = set()
        self.crawl_results = []
    
    async def _setup_browser(self):
        """Setup and configure the browser"""
        self.playwright = await async_playwright().start()
        
        browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--no-sandbox', 
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--allow-running-insecure-content',
                '--disable-features=VizDisplayCompositor'
            ]
        )
        
        context = await browser.new_context(
            viewport={'width': self.viewport_width, 'height': self.viewport_height},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            ignore_https_errors=True,
            bypass_csp=True
        )
        
        # Set timeouts
        context.set_default_timeout(self.page_timeout)
        context.set_default_navigation_timeout(self.navigation_timeout)
        
        return browser, context
    
    async def _cleanup_browser(self, browser: Browser):
        """Properly cleanup browser and playwright resources"""
        try:
            await browser.close()
        except Exception as e:
            print(f"Error closing browser: {e}")
        
        try:
            await self.playwright.stop()
        except Exception as e:
            print(f"Error stopping playwright: {e}")
    
    def _sanitize_filename(self, url: str, suffix: str = "") -> str:
        """Create a safe filename from URL"""
        parsed = urlparse(url)
        filename = f"{parsed.netloc}{parsed.path}".replace('/', '_').replace(':', '')
        # Remove invalid characters
        invalid_chars = '<>:"|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        if suffix:
            filename = f"{filename}_{suffix}"
        return filename[:100]  # Limit length
    
    async def _find_content_breaks(self, page: Page) -> List[int]:
        """Find natural break points in the page content"""
        try:
            # Get positions of major content elements
            break_points = await page.evaluate("""
                () => {
                    const breakPoints = [];
                    const selectors = [
                        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',  // Headers
                        'section', 'article', 'div.content', 'div.main',  // Sections
                        'hr',  // Horizontal rules
                        '.post', '.article', '.card',  // Common content blocks
                        'p:nth-of-type(10n)',  // Every 10th paragraph
                        'nav', 'footer', 'header'  // Page structure
                    ];
                    
                    selectors.forEach(selector => {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => {
                            const rect = el.getBoundingClientRect();
                            const scrollY = window.pageYOffset || document.documentElement.scrollTop;
                            breakPoints.push(Math.round(rect.top + scrollY));
                        });
                    });
                    
                    // Remove duplicates and sort
                    return [...new Set(breakPoints)].sort((a, b) => a - b);
                }
            """)
            
            return [bp for bp in break_points if bp > 0]
        
        except Exception as e:
            print(f"  Warning: Could not find content breaks: {e}")
            return []

    async def _get_optimal_break_points(self, page: Page, total_height: int) -> List[int]:
        """Calculate optimal break points for splitting screenshots"""
        break_points = [0]  # Always start at top
        
        if not self.split_screenshots or total_height <= self.max_screenshot_height:
            return break_points
        
        # Get content-aware break points
        content_breaks = await self._find_content_breaks(page)
        
        current_y = 0
        while current_y < total_height:
            target_y = current_y + self.max_screenshot_height - self.overlap_pixels
            
            if target_y >= total_height:
                break
            
            # Find the best content break near the target
            best_break = target_y
            for content_break in content_breaks:
                if current_y + 200 < content_break <= target_y + 200:  # Within 200px tolerance
                    best_break = content_break
                elif content_break > target_y + 200:
                    break
            
            # Ensure we don't go backwards or create tiny segments
            if best_break > current_y + 300:  # Minimum 300px segments
                break_points.append(best_break - self.overlap_pixels)
                current_y = best_break - self.overlap_pixels
            else:
                break_points.append(target_y)
                current_y = target_y
        
        return break_points

    async def _capture_page_chunks(self, page: Page, url: str, timestamp: int) -> List[str]:
        """Capture page in intelligent chunks"""
        try:
            # Get total page height
            total_height = await page.evaluate("document.body.scrollHeight")
            print(f"  Total page height: {total_height}px")
            
            # Get optimal break points
            break_points = await self._get_optimal_break_points(page, total_height)
            
            screenshot_paths = []
            
            if len(break_points) <= 1:
                # Single screenshot
                filename = f"{self._sanitize_filename(url)}_{timestamp}.png"
                screenshot_path = self.output_dir / filename
                
                if total_height > self.max_screenshot_height * 2:
                    # Very long page - take viewport screenshot instead
                    await page.screenshot(
                        path=str(screenshot_path),
                        full_page=False,
                        type='png',
                        timeout=30000
                    )
                    print(f"  Captured viewport screenshot (page too long): {screenshot_path}")
                else:
                    await page.screenshot(
                        path=str(screenshot_path),
                        full_page=True,
                        type='png',
                        timeout=30000
                    )
                    print(f"  Captured full page screenshot: {screenshot_path}")
                
                screenshot_paths.append(str(screenshot_path))
            else:
                # Multiple chunks
                print(f"  Splitting into {len(break_points)} chunks")
                
                for i, start_y in enumerate(break_points):
                    end_y = break_points[i + 1] if i + 1 < len(break_points) else total_height
                    chunk_height = min(end_y - start_y + self.overlap_pixels, self.max_screenshot_height)
                    
                    filename = f"{self._sanitize_filename(url, f'chunk_{i+1:02d}')}_{timestamp}.png"
                    screenshot_path = self.output_dir / filename
                    
                    # Scroll to position
                    await page.evaluate(f"window.scrollTo(0, {start_y})")
                    await page.wait_for_timeout(500)  # Wait for scroll
                    
                    # Take screenshot of this chunk
                    await page.screenshot(
                        path=str(screenshot_path),
                        type='png',
                        clip={
                            'x': 0,
                            'y': 0,
                            'width': min(self.viewport_width, self.max_screenshot_width),
                            'height': min(chunk_height, self.max_screenshot_height)
                        },
                        timeout=30000
                    )
                    
                    screenshot_paths.append(str(screenshot_path))
                    print(f"  Captured chunk {i+1}: {screenshot_path} (y: {start_y}-{end_y})")
            
            return screenshot_paths
            
        except Exception as e:
            print(f"  Error capturing page chunks: {e}")
            # Fallback to single viewport screenshot
            try:
                filename = f"{self._sanitize_filename(url, 'fallback')}_{timestamp}.png"
                screenshot_path = self.output_dir / filename
                
                await page.screenshot(
                    path=str(screenshot_path),
                    full_page=False,
                    type='png',
                    timeout=15000
                )
                return [str(screenshot_path)]
            except Exception as fallback_error:
                print(f"  Fallback screenshot also failed: {fallback_error}")
                return []

    async def _capture_screenshot(self, page: Page, url: str) -> CrawlResult:
        """Capture screenshot of a single page with retry logic"""
        
        for attempt in range(self.retry_attempts + 1):
            try:
                print(f"Attempting to capture {url} (attempt {attempt + 1}/{self.retry_attempts + 1})")
                
                # Multiple navigation strategies
                strategies = [
                    {'wait_until': 'networkidle', 'timeout': self.navigation_timeout},
                    {'wait_until': 'domcontentloaded', 'timeout': self.navigation_timeout // 2},
                    {'wait_until': 'load', 'timeout': self.navigation_timeout // 3}
                ]
                
                response = None
                navigation_error = None
                
                # Try different wait strategies
                for i, strategy in enumerate(strategies):
                    try:
                        print(f"  Navigation strategy {i+1}: {strategy['wait_until']}")
                        response = await page.goto(url, **strategy)
                        break
                    except Exception as e:
                        navigation_error = e
                        print(f"  Navigation strategy {i+1} failed: {e}")
                        if i < len(strategies) - 1:
                            continue
                        else:
                            raise navigation_error
                
                # Check if page loaded successfully
                if response and response.status >= 400:
                    print(f"  HTTP error: {response.status}")
                    if response.status >= 500 and attempt < self.retry_attempts:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                
                # Wait for page to stabilize
                try:
                    # Try to wait for network idle first
                    await page.wait_for_load_state('networkidle', timeout=5000)
                except:
                    # If that fails, just wait for DOM content
                    try:
                        await page.wait_for_load_state('domcontentloaded', timeout=3000)
                    except:
                        print("  Warning: Could not wait for page load state")
                
                # Additional wait time for dynamic content
                await page.wait_for_timeout(self.wait_time)
                
                # Get page title and height
                try:
                    title = await page.title()
                    total_height = await page.evaluate("document.body.scrollHeight")
                except:
                    title = "Unknown Title"
                    total_height = self.viewport_height
                
                # Create timestamp
                timestamp = int(time.time())
                
                # Capture screenshots (single or multiple chunks)
                screenshot_paths = await self._capture_page_chunks(page, url, timestamp)
                
                if not screenshot_paths:
                    raise Exception("No screenshots were captured")
                
                print(f"  Successfully captured {len(screenshot_paths)} screenshot(s)")
                
                return CrawlResult(
                    url=url,
                    screenshot_paths=screenshot_paths,
                    status_code=response.status if response else 200,
                    title=title,
                    timestamp=timestamp,
                    total_height=total_height,
                    page_count=len(screenshot_paths)
                )
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed for {url}: {str(e)}"
                print(f"  {error_msg}")
                
                if attempt < self.retry_attempts:
                    wait_time = (2 ** attempt) + 1  # Exponential backoff
                    print(f"  Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed
                    return CrawlResult(
                        url=url,
                        screenshot_paths=[],
                        status_code=0,
                        title="",
                        timestamp=time.time(),
                        total_height=0,
                        page_count=0,
                        error=f"All {self.retry_attempts + 1} attempts failed. Last error: {str(e)}"
                    )
    
    async def _extract_links(self, page: Page, base_url: str) -> List[str]:
        """Extract links from the current page"""
        try:
            # Get all links
            links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => link.href).filter(href => href.startsWith('http'));
                }
            """)
            
            # Filter links to same domain
            base_domain = urlparse(base_url).netloc
            same_domain_links = []
            
            for link in links:
                link_domain = urlparse(link).netloc
                if link_domain == base_domain and link not in self.visited_urls:
                    same_domain_links.append(link)
            
            return same_domain_links[:self.max_pages - len(self.visited_urls)]
            
        except Exception as e:
            print(f"Error extracting links: {e}")
            return []
    
    async def crawl_single_page(self, url: str) -> CrawlResult:
        """Crawl and capture screenshot of a single page"""
        browser, context = await self._setup_browser()
        
        try:
            page = await context.new_page()
            result = await self._capture_screenshot(page, url)
            self.crawl_results.append(result)
            return result
            
        finally:
            await self._cleanup_browser(browser)
    
    async def crawl_website(self, start_url: str, crawl_links: bool = True) -> List[CrawlResult]:
        """
        Crawl a website and capture screenshots
        
        Args:
            start_url: Starting URL to crawl
            crawl_links: Whether to follow links and crawl additional pages
            
        Returns:
            List of CrawlResult objects
        """
        browser, context = await self._setup_browser()
        
        try:
            page = await context.new_page()
            
            # URLs to process
            urls_to_process = [start_url]
            
            while urls_to_process and len(self.visited_urls) < self.max_pages:
                current_url = urls_to_process.pop(0)
                
                if current_url in self.visited_urls:
                    continue
                
                print(f"Capturing: {current_url}")
                self.visited_urls.add(current_url)
                
                # Capture screenshot
                result = await self._capture_screenshot(page, current_url)
                self.crawl_results.append(result)
                
                # Extract links if crawling is enabled
                if crawl_links and not result.error:
                    try:
                        new_links = await self._extract_links(page, start_url)
                        urls_to_process.extend(new_links)
                    except Exception as e:
                        print(f"Error extracting links from {current_url}: {e}")
                
                # Small delay between requests
                await asyncio.sleep(1)
            
            return self.crawl_results
            
        finally:
            await self._cleanup_browser(browser)
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of crawl results"""
        successful = [r for r in self.crawl_results if not r.error]
        failed = [r for r in self.crawl_results if r.error]
        
        # Analyze failure reasons
        timeout_failures = [r for r in failed if 'timeout' in str(r.error).lower()]
        http_failures = [r for r in failed if any(code in str(r.error) for code in ['404', '500', '403', '502', '503'])]
        other_failures = [r for r in failed if r not in timeout_failures and r not in http_failures]
        
        # Calculate total screenshots and average chunks
        total_screenshots = sum(len(r.screenshot_paths) for r in successful)
        avg_chunks_per_page = total_screenshots / len(successful) if successful else 0
        
        # Create page details
        page_details = []
        for r in self.crawl_results:
            page_details.append({
                'url': r.url,
                'chunks': len(r.screenshot_paths),
                'height': r.total_height,
                'status': 'success' if not r.error else 'failed'
            })
        
        return {
            'total_pages': len(self.crawl_results),
            'successful': len(successful),
            'failed': len(failed),
            'timeout_failures': len(timeout_failures),
            'http_failures': len(http_failures),
            'other_failures': len(other_failures),
            'output_directory': str(self.output_dir),
            'total_screenshots': total_screenshots,
            'avg_chunks_per_page': avg_chunks_per_page,
            'page_details': page_details,
            'failed_urls': [r.url for r in failed]
        }
    
    def clear_results(self):
        """Clear previous crawl results"""
        self.visited_urls.clear()
        self.crawl_results.clear()


# Example usage and utility functions
async def quick_screenshot(url: str, output_path: str = None, split_mode: bool = True) -> List[str]:
    """
    Quick function to capture screenshots (single or split)
    
    Args:
        url: URL to capture
        output_path: Optional custom output directory
        split_mode: Whether to split long pages
        
    Returns:
        List of paths to the saved screenshots
    """
    output_dir = output_path if output_path else "quick_screenshots"
    crawler = WebsiteVisualCrawler(
        output_dir=output_dir,
        split_screenshots=split_mode,
        max_screenshot_height=1080 if split_mode else 10000
    )
    
    try:
        result = await crawler.crawl_single_page(url)
        return result.screenshot_paths if result.screenshot_paths else []
    
    except Exception as e:
        print(f"Error in quick_screenshot: {e}")
        return []


async def main():
    """Example usage of the WebsiteVisualCrawler"""
    
    try:
        # Initialize crawler with intelligent splitting
        crawler = WebsiteVisualCrawler(
            output_dir="website_screenshots",
            viewport_width=1920,
            viewport_height=1080,
            wait_time=3000,
            max_pages=5,
            split_screenshots=True,
            max_screenshot_height=1080,  # LLM-friendly size
            overlap_pixels=100  # Ensure no content is cut off
        )
        
        # Example 1: Crawl a single page with splitting
        print("=== Single Page Screenshot (with splitting) ===")
        result = await crawler.crawl_single_page("https://www.globalknowledgetech.com")
        print(f"Captured {result.page_count} screenshot chunks")
        for i, path in enumerate(result.screenshot_paths):
            print(f"  Chunk {i+1}: {path}")
        
        # Clear results for next example
        crawler.clear_results()
        
        # Example 2: Crawl website with links
        print("\n=== Website Crawl ===")
        results = await crawler.crawl_website("https://www.reva.edu.in/leadership-management", crawl_links=True)
        
        # Print detailed summary
        summary = crawler.get_results_summary()
        print(f"Crawled {summary['total_pages']} pages")
        print(f"Total screenshots: {summary['total_screenshots']}")
        print(f"Average chunks per page: {summary['avg_chunks_per_page']:.1f}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")   
        print(f"Screenshots saved in: {summary['output_directory']}")
        
        # Show page details
        for page_detail in summary['page_details']:
            print(f"  {page_detail['url']}: {page_detail['chunks']} chunks ({page_detail['height']}px)")
        
        # Example 3: Quick screenshot utility
        print("\n=== Quick Screenshot (Split Mode) ===")
        screenshot_paths = await quick_screenshot("https://httpbin.org/html", split_mode=True)
        print(f"Quick screenshots: {len(screenshot_paths)} files")
        for path in screenshot_paths:
            print(f"  {path}")
        
        # Example 4: Different size configurations
        print("\n=== Custom Size Configuration ===")
        
        # For square format (good for some LLMs)
        square_crawler = WebsiteVisualCrawler(
            output_dir="square_screenshots",
            max_screenshot_width=1000,
            max_screenshot_height=1000,
            split_screenshots=True,
            overlap_pixels=50
        )
        
        result = await square_crawler.crawl_single_page("https://example.com")
        print(f"Square format: {result.page_count} chunks (1000x1000)")
        
    except Exception as e:
        print(f"Error in main: {e}")
    
    finally:
        # Give a moment for cleanup
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    # For Windows, set the event loop policy to avoid issues
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())