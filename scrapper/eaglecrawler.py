import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import Set, List, Dict, Optional, Union
from collections import deque
import hashlib
import re
from datetime import datetime
import os
from enum import Enum

# Playwright imports (optional)
try:
    from playwright.sync_api import sync_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not installed. JavaScript rendering will not be available.")
    print("Install with: pip install playwright && playwright install chromium")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RenderMode(Enum):
    """Enumeration for different rendering modes"""
    STATIC = "static"           # Standard HTTP request + BeautifulSoup
    JAVASCRIPT = "javascript"   # Full browser rendering with Playwright

@dataclass
class CrawlConfig:
    """Configuration for individual URL crawling"""
    url: str
    render_mode: RenderMode = RenderMode.STATIC
    wait_time: float = 3.0              # Time to wait for JS to load (seconds)
    wait_for_selector: Optional[str] = None  # CSS selector to wait for
    execute_js: Optional[str] = None    # Custom JavaScript to execute
    screenshot: bool = False            # Whether to take a screenshot
    block_resources: List[str] = None   # Resource types to block (images, stylesheets, etc.)

@dataclass
class CrawlResult:
    """Structure to hold crawled page data"""
    url: str
    title: str
    content: str
    links: List[str]
    metadata: Dict
    crawl_timestamp: str
    content_hash: str
    render_mode: str
    screenshot_path: Optional[str] = None

class JavaScriptRenderer:
    """Handles JavaScript rendering using Playwright"""
    
    def __init__(self, headless: bool = True, user_agent: str = None):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is not installed. Install with: pip install playwright")
        
        self.headless = headless
        self.user_agent = user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        self.playwright = None
        self.browser = None
    
    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
    
    def render_page(self, config: CrawlConfig) -> Dict[str, str]:
        """
        Render a page with JavaScript and return HTML content
        
        Returns:
            Dict with 'html' and optional 'screenshot_path'
        """
        if not self.browser:
            raise RuntimeError("Browser not initialized. Use within context manager.")
        
        page = self.browser.new_page(user_agent=self.user_agent)
        result = {'html': '', 'screenshot_path': None}
        
        try:
            # Block resources if specified
            if config.block_resources:
                def handle_route(route):
                    if route.request.resource_type in config.block_resources:
                        route.abort()
                    else:
                        route.continue_()
                page.route("**/*", handle_route)
            
            # Navigate to page
            page.goto(config.url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait for specific selector if provided
            if config.wait_for_selector:
                page.wait_for_selector(config.wait_for_selector, timeout=10000)
            
            # Wait for additional time for JS to execute
            if config.wait_time > 0:
                page.wait_for_timeout(int(config.wait_time * 1000))
            
            # Execute custom JavaScript if provided
            if config.execute_js:
                page.evaluate(config.execute_js)
                page.wait_for_timeout(1000)  # Brief wait after JS execution
            
            # Get final HTML content
            result['html'] = page.content()
            
            # Take screenshot if requested
            if config.screenshot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_url = re.sub(r'[^\w\-_.]', '_', config.url)
                screenshot_path = f"screenshots/{safe_url}_{timestamp}.png"
                os.makedirs('screenshots', exist_ok=True)
                page.screenshot(path=screenshot_path, full_page=True)
                result['screenshot_path'] = screenshot_path
                logger.info(f"Screenshot saved: {screenshot_path}")
            
        except Exception as e:
            logger.error(f"Error rendering {config.url}: {str(e)}")
            raise
        finally:
            page.close()
        
        return result

class WebCrawler:
    def __init__(self, 
                 max_depth: int = 3,
                 delay: float = 1.0,
                 max_pages: int = 100,
                 respect_robots: bool = True,
                 allowed_domains: Optional[List[str]] = None,
                 exclude_patterns: Optional[List[str]] = None,
                 js_render_headless: bool = True):
        """
        Initialize the web crawler with optional JavaScript rendering
        
        Args:
            max_depth: Maximum crawl depth
            delay: Delay between requests (seconds)
            max_pages: Maximum number of pages to crawl
            respect_robots: Whether to respect robots.txt
            allowed_domains: List of allowed domains (None = allow all)
            exclude_patterns: List of regex patterns to exclude URLs
            js_render_headless: Whether to run browser in headless mode
        """
        self.max_depth = max_depth
        self.delay = delay
        self.max_pages = max_pages
        self.respect_robots = respect_robots
        self.allowed_domains = allowed_domains or []
        self.exclude_patterns = [re.compile(pattern) for pattern in (exclude_patterns or [])]
        self.js_render_headless = js_render_headless
        
        # Internal state
        self.visited_urls: Set[str] = set()
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.crawl_results: List[CrawlResult] = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Python Web Crawler 1.0 (Educational Purpose)'
        })
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and query parameters"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    def _is_allowed_domain(self, url: str) -> bool:
        """Check if URL domain is in allowed domains"""
        if not self.allowed_domains:
            return True
        domain = urlparse(url).netloc
        return any(allowed in domain for allowed in self.allowed_domains)
    
    def _is_excluded_pattern(self, url: str) -> bool:
        """Check if URL matches any exclude patterns"""
        return any(pattern.search(url) for pattern in self.exclude_patterns)
    
    def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        if not self.respect_robots:
            return True
        
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if base_url not in self.robots_cache:
            robots_url = urljoin(base_url, '/robots.txt')
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
                self.robots_cache[base_url] = rp
            except:
                # If robots.txt can't be read, assume allowed
                self.robots_cache[base_url] = None
        
        robots_parser = self.robots_cache[base_url]
        if robots_parser is None:
            return True
        
        return robots_parser.can_fetch(self.session.headers['User-Agent'], url)
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML, removing scripts and styles"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the page"""
        links = []
        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(base_url, link['href'])
            normalized_url = self._normalize_url(absolute_url)
            
            # Basic filtering
            if (normalized_url.startswith(('http://', 'https://')) and 
                normalized_url not in self.visited_urls and
                self._is_allowed_domain(normalized_url) and
                not self._is_excluded_pattern(normalized_url)):
                links.append(normalized_url)
        
        return links
    
    def _crawl_page_static(self, url: str) -> Optional[str]:
        """Crawl a page using standard HTTP request"""
        try:
            if not self._check_robots_txt(url):
                logger.info(f"Skipping {url} due to robots.txt")
                return None
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type.lower():
                logger.info(f"Skipping non-HTML content: {url}")
                return None
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error crawling {url} (static): {str(e)}")
            return None
    
    def _crawl_page_with_config(self, config: CrawlConfig) -> Optional[CrawlResult]:
        """Crawl a single page using the specified configuration"""
        html_content = None
        screenshot_path = None
        
        try:
            if config.render_mode == RenderMode.STATIC:
                html_content = self._crawl_page_static(config.url)
            
            elif config.render_mode == RenderMode.JAVASCRIPT:
                if not PLAYWRIGHT_AVAILABLE:
                    logger.warning(f"Playwright not available, falling back to static for {config.url}")
                    html_content = self._crawl_page_static(config.url)
                else:
                    with JavaScriptRenderer(headless=self.js_render_headless) as renderer:
                        result = renderer.render_page(config)
                        html_content = result['html']
                        screenshot_path = result.get('screenshot_path')
            
            if not html_content:
                return None
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract data
            title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
            content = self._extract_content(soup)
            links = self._extract_links(soup, config.url)
            
            # Create content hash for deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Metadata
            metadata = {
                'content_length': len(content),
                'num_links': len(links),
                'domain': urlparse(config.url).netloc,
                'render_mode': config.render_mode.value
            }
            
            return CrawlResult(
                url=config.url,
                title=title,
                content=content,
                links=links,
                metadata=metadata,
                crawl_timestamp=datetime.now().isoformat(),
                content_hash=content_hash,
                render_mode=config.render_mode.value,
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            logger.error(f"Error crawling {config.url}: {str(e)}")
            return None
    
    def crawl_with_configs(self, crawl_configs: List[CrawlConfig]) -> List[CrawlResult]:
        """
        Crawl pages using specific configurations for each URL
        
        Args:
            crawl_configs: List of CrawlConfig objects specifying how to crawl each URL
        """
        logger.info(f"Starting crawl with {len(crawl_configs)} configured URLs")
        
        for config in crawl_configs:
            if len(self.visited_urls) >= self.max_pages:
                break
            
            if config.url in self.visited_urls:
                continue
            
            logger.info(f"Crawling ({config.render_mode.value}): {config.url}")
            
            # Mark as visited
            self.visited_urls.add(config.url)
            
            # Crawl the page
            result = self._crawl_page_with_config(config)
            
            if result:
                self.crawl_results.append(result)
            
            # Respect rate limiting
            time.sleep(self.delay)
        
        logger.info(f"Crawl completed. Processed {len(crawl_configs)} URLs, "
                   f"extracted {len(self.crawl_results)} results")
        
        return self.crawl_results
    
    def crawl(self, start_urls: List[str]) -> List[CrawlResult]:
        """
        Standard crawling method (backward compatibility)
        Uses static rendering for all URLs
        """
        configs = [CrawlConfig(url=url, render_mode=RenderMode.STATIC) for url in start_urls]
        return self.crawl_with_configs(configs)
    
    def save_results(self, filename: str):
        """Save crawl results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([asdict(result) for result in self.crawl_results], 
                     f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filename}")
    
    def search_content(self, query: str, limit: int = 10) -> List[Dict]:
        """Simple text search across crawled content"""
        query_lower = query.lower()
        matches = []
        
        for result in self.crawl_results:
            content_lower = result.content.lower()
            if query_lower in content_lower:
                # Find snippet around the match
                match_pos = content_lower.find(query_lower)
                start = max(0, match_pos - 100)
                end = min(len(result.content), match_pos + 100)
                snippet = result.content[start:end]
                
                matches.append({
                    'url': result.url,
                    'title': result.title,
                    'snippet': snippet,
                    'relevance_score': content_lower.count(query_lower),
                    'render_mode': result.render_mode
                })
        
        # Sort by relevance
        matches.sort(key=lambda x: x['relevance_score'], reverse=True)
        return matches[:limit]

# Helper functions for easier configuration
def create_static_config(url: str) -> CrawlConfig:
    """Create a static rendering configuration"""
    return CrawlConfig(url=url, render_mode=RenderMode.STATIC)

def create_js_config(url: str, 
                    wait_time: float = 3.0,
                    wait_for_selector: str = None,
                    execute_js: str = None,
                    screenshot: bool = False,
                    block_resources: List[str] = None) -> CrawlConfig:
    """Create a JavaScript rendering configuration"""
    return CrawlConfig(
        url=url,
        render_mode=RenderMode.JAVASCRIPT,
        wait_time=wait_time,
        wait_for_selector=wait_for_selector,
        execute_js=execute_js,
        screenshot=screenshot,
        block_resources=block_resources or []
    )

# Example usage
def main():
    """Example usage demonstrating mixed rendering modes"""
    
    crawler = WebCrawler(
        max_depth=1,
        delay=2.0,  # Longer delay for JS rendering
        max_pages=10,
        js_render_headless=True
    )
    
    # Configure different rendering modes for different URLs
    crawl_configs = [
        # Static page (fast)
        create_static_config('https://example.com'),
        
        # JavaScript-heavy SPA with custom wait
        create_js_config(
            'https://example-spa.com',
            wait_time=5.0,
            wait_for_selector='.main-content',
            screenshot=True,
            block_resources=['image', 'stylesheet']  # Speed up loading
        ),
        
        # Page requiring custom JavaScript execution
        create_js_config(
            'https://example-interactive.com',
            execute_js='document.querySelector(".load-more").click();',
            wait_time=3.0
        ),
        
        # Standard JS rendering
        create_js_config('https://react-app.com', wait_time=4.0)
    ]
    
    # Crawl with mixed configurations
    results = crawler.crawl_with_configs(crawl_configs)
    
    # Save results
    crawler.save_results('mixed_crawl_results.json')
    
    # Show results summary
    for result in results:
        print(f"URL: {result.url}")
        print(f"Render Mode: {result.render_mode}")
        print(f"Title: {result.title}")
        print(f"Content Length: {len(result.content)}")
        if result.screenshot_path:
            print(f"Screenshot: {result.screenshot_path}")
        print("-" * 50)

if __name__ == "__main__":
    main()