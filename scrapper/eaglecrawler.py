"""
Enhanced Website Crawler - A Playwright-based module for intelligent visual and text crawling
"""

import asyncio
import os
import time
import json
import math
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, asdict
import hashlib
import re
from collections import defaultdict

from playwright.async_api import async_playwright, Page, Browser, BrowserContext


@dataclass
class CrawlResult:
    """Data class to store crawl results
    
    Attributes:
        url (str): URL of the crawled page
        screenshot_paths (List[str]): Paths to saved screenshots (visual mode)
        text_content (str): Extracted text content (text mode)
        html_content (str): Raw HTML content
        status_code (int): HTTP status code of the page
        title (str): Page title
        timestamp (float): Unix timestamp of crawl time
        total_height (int): Total scroll height of the page in pixels
        page_count (int): Number of pages captured (for visual mode)
        links (List[str]): List of URLs found on the page
        metadata (Dict[str, Any]): Extracted metadata and additional information
        error (Optional[str]): Error message if crawl failed
        mode (str): Crawling mode used ('visual', 'text', or 'both')
    """
    url: str
    screenshot_paths: List[str] = None
    text_content: str = ""
    html_content: str = ""
    status_code: int = 0
    title: str = ""
    timestamp: float = 0
    total_height: int = 0
    page_count: int = 0
    links: List[str] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    mode: str = "visual"

    def __post_init__(self):
        """Initialize default values for mutable attributes"""
        if self.screenshot_paths is None:
            self.screenshot_paths = []
        if self.links is None:
            self.links = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class A4Dimensions:
    """A4 page dimensions in pixels at different DPIs
    
    Class Attributes:
        WIDTH_72DPI (int): Width at 72 DPI
        HEIGHT_72DPI (int): Height at 72 DPI
        WIDTH_96DPI (int): Width at 96 DPI
        HEIGHT_96DPI (int): Height at 96 DPI
        WIDTH_150DPI (int): Width at 150 DPI
        HEIGHT_150DPI (int): Height at 150 DPI
        WIDTH_DESKTOP (int): Desktop viewport width
        HEIGHT_DESKTOP (int): Desktop viewport height
    """
    WIDTH_72DPI = 595
    HEIGHT_72DPI = 842
    WIDTH_96DPI = 794
    HEIGHT_96DPI = 1123
    WIDTH_150DPI = 1240
    HEIGHT_150DPI = 1754
    WIDTH_DESKTOP = 1523
    HEIGHT_DESKTOP = 2707
    
    @classmethod
    def get_dimensions(cls, width: int = 1920) -> Tuple[int, int]:
        """Get A4 dimensions for specified DPI
        
        Args:
            dpi: Dots per inch resolution
            
        Returns:
            Tuple of (width, height) in pixels for the specified DPI
            
        Notes:
            - Uses 96 DPI as default
            - For DPI > 150, returns desktop dimensions
        """
        if width > 0:
            return width, round(width*1.4143576826196473551637279596977)
        else:
            raise Exception("Page width should be higher than 0!")


class ContentAnalyzer:
    """Analyzes page content to determine intelligent break points"""
    
    @staticmethod
    async def get_content_sections(page: Page) -> List[Dict[str, Any]]:
        """Get semantic content sections with priorities from page
        
        Args:
            page: Playwright Page instance
            
        Returns:
            List of dictionaries containing section details:
            - selector: CSS selector
            - element_index: Index in DOM
            - priority: Importance level (1-10)
            - type: Content type (header, media, etc)
            - minBuffer: Minimum buffer space needed (px)
            - top: Absolute top position (px)
            - bottom: Absolute bottom position (px)
            - height: Element height (px)
            - width: Element width (px)
            - centerY: Vertical center point (px)
            - tagName: HTML tag name
            - className: CSS class(es)
            - id: Element ID
            - textContent: First 100 characters of text
            - hasImages: Whether element contains images
            - hasText: Whether element contains significant text
            
        Notes:
            - Filters out small/invisible elements
            - Sorts by vertical position and priority
        """
        try:
            sections = await page.evaluate("""
                () => {
                    const sections = [];
                    
                    // Define element priorities and types
                    const elementConfig = {
                        // High priority - never split
                        'h1': { priority: 10, type: 'header', minBuffer: 50 },
                        'h2': { priority: 9, type: 'header', minBuffer: 40 },
                        'h3': { priority: 8, type: 'header', minBuffer: 30 },
                        'h4': { priority: 7, type: 'header', minBuffer: 20 },
                        'h5': { priority: 6, type: 'header', minBuffer: 20 },
                        'h6': { priority: 5, type: 'header', minBuffer: 20 },
                        
                        // Medium-high priority
                        'figure': { priority: 8, type: 'media', minBuffer: 30 },
                        'img': { priority: 7, type: 'media', minBuffer: 20 },
                        'video': { priority: 8, type: 'media', minBuffer: 30 },
                        'canvas': { priority: 7, type: 'media', minBuffer: 20 },
                        'svg': { priority: 6, type: 'media', minBuffer: 15 },
                        
                        // Medium priority
                        'table': { priority: 6, type: 'data', minBuffer: 25 },
                        'form': { priority: 6, type: 'interactive', minBuffer: 25 },
                        'blockquote': { priority: 5, type: 'content', minBuffer: 20 },
                        'pre': { priority: 5, type: 'code', minBuffer: 15 },
                        'code': { priority: 4, type: 'code', minBuffer: 10 },
                        
                        // Lower priority but still important
                        'article': { priority: 4, type: 'container', minBuffer: 20 },
                        'section': { priority: 3, type: 'container', minBuffer: 15 },
                        'div.card': { priority: 4, type: 'component', minBuffer: 15 },
                        'div.post': { priority: 4, type: 'component', minBuffer: 15 },
                        'div.article': { priority: 4, type: 'component', minBuffer: 15 },
                        
                        // Navigation and structure
                        'nav': { priority: 3, type: 'navigation', minBuffer: 10 },
                        'header': { priority: 3, type: 'structure', minBuffer: 15 },
                        'footer': { priority: 2, type: 'structure', minBuffer: 10 },
                        'aside': { priority: 2, type: 'sidebar', minBuffer: 10 },
                        
                        // Lists
                        'ul': { priority: 3, type: 'list', minBuffer: 10 },
                        'ol': { priority: 3, type: 'list', minBuffer: 10 },
                        'dl': { priority: 3, type: 'list', minBuffer: 10 },
                        
                        // Text blocks
                        'p': { priority: 2, type: 'text', minBuffer: 5 }
                    };
                    
                    // Get all selectors
                    const selectors = Object.keys(elementConfig);
                    
                    selectors.forEach(selector => {
                        const config = elementConfig[selector];
                        const elements = document.querySelectorAll(selector);
                        
                        elements.forEach((el, index) => {
                            const rect = el.getBoundingClientRect();
                            const scrollY = window.pageYOffset || document.documentElement.scrollTop;
                            
                            // Filter out tiny or invisible elements
                            if (rect.height < 10 || rect.width < 10 || 
                                rect.top + scrollY < 0 || 
                                getComputedStyle(el).display === 'none') {
                                return;
                            }
                            
                            const absoluteTop = Math.round(rect.top + scrollY);
                            const absoluteBottom = Math.round(rect.bottom + scrollY);
                            
                            sections.push({
                                selector: selector,
                                element_index: index,
                                priority: config.priority,
                                type: config.type,
                                minBuffer: config.minBuffer,
                                top: absoluteTop,
                                bottom: absoluteBottom,
                                height: Math.round(rect.height),
                                width: Math.round(rect.width),
                                centerY: Math.round(absoluteTop + rect.height / 2),
                                tagName: el.tagName.toLowerCase(),
                                className: el.className || '',
                                id: el.id || '',
                                textContent: el.textContent ? el.textContent.substring(0, 100) : '',
                                hasImages: el.querySelectorAll('img').length > 0,
                                hasText: (el.textContent || '').trim().length > 20
                            });
                        });
                    });
                    
                    // Sort by position first, then by priority
                    return sections.sort((a, b) => {
                        if (Math.abs(a.top - b.top) < 20) {
                            return b.priority - a.priority; // Higher priority first if close in position
                        }
                        return a.top - b.top; // Position first
                    });
                }
            """)
            
            return sections
            
        except Exception as e:
            print(f"  Warning: Could not analyze content sections: {e}")
            return []
    
    @staticmethod
    def find_safe_break_points(sections: List[Dict[str, Any]], 
                              total_height: int, 
                              page_height: int,
                              min_overlap: int = 50) -> List[Dict[str, Any]]:
        """Find safe break points that don't cut important content
        
        Args:
            sections: Content sections from get_content_sections()
            total_height: Full page height in pixels
            page_height: Viewport height in pixels
            min_overlap: Minimum page overlap in pixels
            
        Returns:
            List of break point dictionaries containing:
            - y: Scroll position (px)
            - height: Chunk height (px)
            - type: Break type ('full_page', 'last_chunk', 'content_chunk')
            - chunk_index: Page number index
            - elements_preserved: Count of preserved elements
            - break_reason: Description of why break was chosen
            - next_start_y: Next starting position (for multi-page)
            
        Notes:
            - Uses multiple strategies to avoid cutting important content
            - Enforces minimum/maximum chunk sizes
        """
        
        if total_height <= page_height:
            return [{
                'y': 0,
                'height': total_height,
                'type': 'full_page',
                'chunk_index': 1,
                'elements_preserved': len(sections)
            }]
        
        break_points = []
        current_y = 0
        chunk_index = 0
        
        while current_y < total_height:
            chunk_index += 1
            ideal_end_y = current_y + page_height
            
            if ideal_end_y >= total_height:
                # Last chunk
                actual_start_y = max(0, current_y - min_overlap)
                break_points.append({
                    'y': actual_start_y,
                    'height': total_height - actual_start_y,
                    'type': 'last_chunk',
                    'chunk_index': chunk_index,
                    'elements_preserved': 0,
                    'break_reason': 'end_of_page'
                })
                break
            
            # Find the best break point near the ideal position
            best_break_y = ideal_end_y
            break_reason = 'default_split'
            elements_preserved = 0
            
            # Look for safe break areas (±200px from ideal)
            search_start = max(current_y + page_height // 2, ideal_end_y - 200)
            search_end = min(total_height, ideal_end_y + 200)
            
            # Find elements that would be affected by breaking at ideal position
            affected_elements = [
                s for s in sections 
                if s['top'] < ideal_end_y < s['bottom'] and s['height'] > 30
            ]
            
            if affected_elements:
                # Sort affected elements by priority (higher first)
                affected_elements.sort(key=lambda x: (-x['priority'], x['height']))
                
                for element in affected_elements:
                    # Strategy 1: Break before high-priority elements
                    if element['priority'] >= 7:  # Headers, important media
                        break_before_y = element['top'] - element['minBuffer']
                        if search_start <= break_before_y <= search_end and break_before_y > current_y + 300:
                            best_break_y = break_before_y
                            break_reason = f'before_priority_{element["priority"]}_{element["type"]}'
                            elements_preserved += 1
                            break
                    
                    # Strategy 2: Break after complete elements
                    elif element['priority'] >= 4:
                        break_after_y = element['bottom'] + element['minBuffer']
                        if search_start <= break_after_y <= search_end:
                            best_break_y = break_after_y
                            break_reason = f'after_{element["type"]}'
                            elements_preserved += 1
                            break
            
            # Strategy 3: Find natural content gaps
            if best_break_y == ideal_end_y:
                # Look for gaps between elements
                for i in range(len(sections) - 1):
                    current_elem = sections[i]
                    next_elem = sections[i + 1]
                    
                    gap_start = current_elem['bottom']
                    gap_end = next_elem['top']
                    gap_size = gap_end - gap_start
                    
                    # Look for significant gaps (>30px) in search area
                    if (gap_size > 30 and 
                        search_start <= gap_start <= search_end and
                        gap_start > current_y + 300):
                        
                        best_break_y = gap_start + gap_size // 2
                        break_reason = f'content_gap_{gap_size}px'
                        break
            
            # Ensure reasonable chunk size
            if best_break_y - current_y < page_height * 0.4:  # Too small
                best_break_y = current_y + int(page_height * 0.6)
                break_reason = 'min_size_enforced'
            elif best_break_y - current_y > page_height * 1.5:  # Too large
                best_break_y = current_y + int(page_height * 1.2)
                break_reason = 'max_size_enforced'
            
            # Add the break point
            actual_start_y = max(0, current_y - (min_overlap if chunk_index > 1 else 0))
            chunk_height = best_break_y - actual_start_y + min_overlap
            
            break_points.append({
                'y': actual_start_y,
                'height': min(chunk_height, page_height),
                'type': 'content_chunk',
                'chunk_index': chunk_index,
                'break_reason': break_reason,
                'elements_preserved': elements_preserved,
                'next_start_y': best_break_y - min_overlap
            })
            
            current_y = best_break_y - min_overlap
        
        return break_points


class EnhancedCrawler:
    """
    Enhanced website crawler with visual/text modes and page splitting
    
    Attributes:
        mode: 'visual', 'text', or 'both'
        output_dir: Output directory path
        page_width: DPI for A4 sizing (72, 96, 150)
        min_overlap: Minimum page overlap in pixels
        smart_splitting: Enable content-aware splitting
        preserve_context: Prevent cutting important elements
        wait_time: Wait time before capture (ms)
        headless: Run browser headlessly
        max_pages: Maximum pages to crawl
        page_timeout: Page load timeout (ms)
        navigation_timeout: Navigation timeout (ms)
        retry_attempts: Retry attempts for failures
        extract_links: Extract hyperlinks from pages
        extract_images: Extract image metadata
        clean_text: Clean extracted text content
        save_html: Save raw HTML content
        content_selectors: CSS selectors for content extraction
        max_depth: Maximum depth to crawl (1 = no recursion, 2 = one level deep, etc.)
        same_domain_only: Only crawl URLs from the same domain as starting URLs
        url_patterns: List of regex patterns that URLs must match to be crawled
        exclude_patterns: List of regex patterns to exclude from crawling
        delay_between_requests: Delay in seconds between requests to be respectful
        min_content_length: Minimum character threshold for content detection
    """
    
    def __init__(
        self,
        mode: str = "visual",
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
        Initialize the enhanced crawler
        
        Args:
            mode: Crawling mode - "visual", "text", or "both"
            output_dir: Directory to save outputs
            page_width: DPI setting for A4 page dimensions (72, 96, or 150)
            min_overlap: Minimum overlap between page chunks in pixels
            smart_splitting: Use intelligent content-aware splitting
            preserve_context: Ensure important elements aren't cut off
            wait_time: Time to wait before capture/extraction (ms)
            headless: Run browser in headless mode
            max_pages: Maximum number of pages to crawl
            page_timeout: Overall page timeout (ms)
            navigation_timeout: Navigation timeout (ms)
            retry_attempts: Number of retry attempts for failed pages
            extract_links: Extract all links from pages
            extract_images: Extract image information
            clean_text: Clean and format extracted text
            save_html: Save raw HTML content
            content_selectors: Custom CSS selectors for content extraction
            min_content_length: Minimum character threshold for content detection
        """
        self.mode = mode.lower()
        if self.mode not in ["visual", "text", "both"]:
            raise ValueError("Mode must be 'visual', 'text', or 'both'")
            
        self.output_dir = Path(output_dir)
        
        # A4 dimensions
        self.a4_width, self.a4_height = A4Dimensions.get_dimensions(page_width)
        self.page_width = page_width
        self.min_overlap = min_overlap
        self.smart_splitting = smart_splitting
        self.preserve_context = preserve_context
        
        # General settings
        self.wait_time = wait_time
        self.headless = headless
        self.max_pages = max_pages
        self.page_timeout = page_timeout
        self.navigation_timeout = navigation_timeout
        self.retry_attempts = retry_attempts
        
        # Text settings
        self.extract_links = extract_links
        self.extract_images = extract_images
        self.clean_text = clean_text
        self.save_html = save_html
        self.content_selectors = content_selectors or [
            'main', 'article', '.content', '.main-content', 
            '.post-content', '.entry-content', '.page-content',
            'section', '.container', 'body'
        ]
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        if self.mode in ["visual", "both"]:
            (self.output_dir / "screenshots").mkdir(exist_ok=True)
        if self.mode in ["text", "both"]:
            (self.output_dir / "text").mkdir(exist_ok=True)
            if self.save_html:
                (self.output_dir / "html").mkdir(exist_ok=True)
        
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.url_patterns = [re.compile(pattern) for pattern in (url_patterns or [])]
        self.exclude_patterns = [re.compile(pattern) for pattern in (exclude_patterns or [])]
        self.delay_between_requests = delay_between_requests
        
        # Content detection settings
        self.min_content_length = min_content_length
        
        # Boilerplate removal settings
        self.boilerplate_shingle_size = boilerplate_shingle_size
        self.boilerplate_threshold = boilerplate_threshold
        
        # Track crawling state
        self.visited_urls = set()
        self.crawl_queue = []  # List of (url, depth) tuples
        self.allowed_domains = set()  # Domains we're allowed to crawl
        self.crawl_results = []
        self.url_to_depth = {}  # Track depth of each URL
        
        print(f"Initialized crawler with A4 dimensions: {self.a4_width}x{self.a4_height}px at {page_width}DPI")
    
    def _is_valid_url(self, url: str, base_url: str, current_depth: int) -> bool:
        """
        Check if a URL should be crawled based on filtering rules
        
        Args:
            url: URL to validate
            base_url: Base URL for resolving relative links
            current_depth: Current crawling depth
            
        Returns:
            True if URL should be crawled, False otherwise
        """
        try:
            # Resolve relative URLs
            absolute_url = urljoin(base_url, url)
            parsed_url = urlparse(absolute_url)
            
            # Skip non-HTTP(S) URLs
            if parsed_url.scheme not in ['http', 'https']:
                return False
            
            # Skip if already visited
            if absolute_url in self.visited_urls:
                return False
            
            # Check depth limit
            if current_depth >= self.max_depth:
                return False
            
            # Check domain restrictions
            if self.same_domain_only and parsed_url.netloc not in self.allowed_domains:
                return False
            
            # Check URL patterns (must match at least one if patterns are specified)
            if self.url_patterns:
                if not any(pattern.search(absolute_url) for pattern in self.url_patterns):
                    return False
            
            # Check exclude patterns (must not match any)
            if self.exclude_patterns:
                if any(pattern.search(absolute_url) for pattern in self.exclude_patterns):
                    return False
            
            # Skip common non-content URLs
            skip_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                             '.zip', '.rar', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif', 
                             '.svg', '.ico', '.css', '.js', '.xml', '.json'}
            
            path_lower = parsed_url.path.lower()
            if any(path_lower.endswith(ext) for ext in skip_extensions):
                return False
            
            # Skip fragments and JavaScript URLs
            if parsed_url.fragment or absolute_url.startswith('javascript:'):
                return False
            
            return True
            
        except Exception as e:
            print(f"  Error validating URL {url}: {e}")
            return False

    def _extract_links_from_result(self, result: CrawlResult, current_depth: int) -> List[str]:
        """
        Extract and filter links from a crawl result for further crawling
        
        Args:
            result: CrawlResult containing extracted links
            current_depth: Current depth in crawl tree
            
        Returns:
            List of valid URLs to crawl next
        """
        valid_links = []
        
        if not result.links or current_depth >= self.max_depth:
            return valid_links
        
        for link in result.links:
            if self._is_valid_url(link, result.url, current_depth + 1):
                absolute_url = urljoin(result.url, link)
                valid_links.append(absolute_url)
        
        return valid_links

    def _initialize_crawl_queue(self, start_urls: List[str]) -> None:
        """
        Initialize the crawl queue with starting URLs
        
        Args:
            start_urls: List of starting URLs
        """
        self.crawl_queue = []
        self.visited_urls = set()
        self.allowed_domains = set()
        self.url_to_depth = {}
        
        # Extract allowed domains from start URLs
        for url in start_urls:
            try:
                parsed = urlparse(url)
                if parsed.netloc:
                    self.allowed_domains.add(parsed.netloc)
                
                # Add to queue with depth 0
                self.crawl_queue.append((url, 0))
                self.url_to_depth[url] = 0
                
            except Exception as e:
                print(f"Error parsing start URL {url}: {e}")
        
        print(f"Initialized crawl queue with {len(self.crawl_queue)} URLs")
        print(f"Allowed domains: {self.allowed_domains}")

    async def _setup_browser(self):
        """Setup and configure the browser with A4 viewport
        
        Returns:
            Tuple of (Browser, BrowserContext) instances
            
        Notes:
            - Launches Chromium browser with security-disabled flags
            - Sets viewport to configured A4 dimensions
            - Applies timeouts and custom user agent
        """
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
            viewport={'width': self.a4_width, 'height': self.a4_height},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            ignore_https_errors=True,
            bypass_csp=True
        )
        
        # Set timeouts
        context.set_default_timeout(self.page_timeout)
        context.set_default_navigation_timeout(self.navigation_timeout)
        
        return browser, context
    
    async def _cleanup_browser(self, browser: Browser):
        """Properly cleanup browser and playwright resources
        
        Args:
            browser: Browser instance to close
        """
        try:
            await browser.close()
        except Exception as e:
            print(f"Error closing browser: {e}")
        
        try:
            await self.playwright.stop()
        except Exception as e:
            print(f"Error stopping playwright: {e}")
    
    def _sanitize_filename(self, url: str, suffix: str = "") -> str:
        """Create a safe filename from URL
        
        Args:
            url: Page URL to sanitize
            suffix: Additional filename suffix
            
        Returns:
            Sanitized filename string
            
        Notes:
            - Removes special characters and limits length
            - Uses domain + path structure"""
        parsed = urlparse(url)
        filename = f"{parsed.netloc}{parsed.path}".replace('/', '_').replace(':', '')
        # Remove invalid characters
        invalid_chars = '<>:"|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        if suffix:
            filename = f"{filename}_{suffix}"
        return filename[:100]  # Limit length

    async def _find_optimal_content_element(self, page: Page) -> Optional[str]:
        """Intelligently find the best content container using heuristics
        
        Args:
            page: Playwright Page instance
            
        Returns:
            HTML string of the best content container, or None if not found
        """
        try:
            return await page.evaluate("""() => {
                // Get all content container candidates
                const candidates = Array.from(document.querySelectorAll(
                    'div, section, article, main, .content, .container, .main, .body'
                )).filter(el => {
                    // Filter out small/invisible elements
                    const rect = el.getBoundingClientRect();
                    return rect.width > 300 && rect.height > 200 && 
                           getComputedStyle(el).display !== 'none';
                });

                if (!candidates.length) return null;

                // Score elements based on content characteristics
                const scored = candidates.map(el => {
                    const rect = el.getBoundingClientRect();
                    const text = el.textContent || '';
                    
                    // Calculate text density score
                    const area = rect.width * rect.height;
                    const density = area > 0 ? text.length / area : 0;
                    
                    // Calculate structural score
                    let structureScore = 0;
                    if (el.tagName === 'ARTICLE') structureScore += 3;
                    if (el.tagName === 'MAIN') structureScore += 2;
                    if (el.classList.contains('content')) structureScore += 2;
                    if (el.classList.contains('main')) structureScore += 1;
                    if (el.classList.contains('container')) structureScore += 1;
                    
                    // Calculate position score (center of viewport)
                    const viewportCenter = window.innerHeight / 2;
                    const elementCenter = rect.top + (rect.height / 2);
                    const positionScore = 1 - Math.abs(viewportCenter - elementCenter) / viewportCenter;
                    
                    // Calculate heading score
                    const headingCount = el.querySelectorAll('h1, h2, h3, h4, h5, h6').length;
                    const headingScore = Math.min(10, headingCount * 2);
                    
                    // Calculate paragraph score
                    const paragraphCount = el.querySelectorAll('p').length;
                    const paragraphScore = Math.min(10, paragraphCount * 0.5);
                    
                    // Final score weighting
                    return {
                        element: el,
                        score: (density * 40) + (structureScore * 20) + 
                               (positionScore * 20) + (headingScore * 10) +
                               (paragraphScore * 10)
                    };
                });

                // Return the highest scoring element
                scored.sort((a, b) => b.score - a.score);
                return scored[0].element.outerHTML;
            }""")
        except Exception as e:
            print(f"Content detection error: {e}")
            return None

    async def _capture_visual_content(self, page: Page, url: str, timestamp: int) -> List[str]:
        """Capture visual content with intelligent A4-sized splitting
        
        Args:
            page: Playwright Page instance
            url: Page URL for naming
            timestamp: Timestamp for unique filenames
            
        Returns:
            List of paths to saved screenshots
            
        Notes:
            - Uses content analysis for intelligent splitting
            - Falls back to single screenshot on error
            - Applies overlap and position adjustments"""
        try:
            # Get total page height and content analysis
            total_height = await page.evaluate("document.body.scrollHeight")
            print(f"  Total page height: {total_height}px")
            print(f"  A4 page size: {self.a4_width}x{self.a4_height}px")
            
            # Analyze content sections if smart splitting is enabled
            if self.smart_splitting and self.preserve_context:
                print("  Analyzing content sections...")
                sections = await ContentAnalyzer.get_content_sections(page)
                print(f"  Found {len(sections)} content sections")
            else:
                sections = []

            # Calculate break points
            break_points = ContentAnalyzer.find_safe_break_points(
                sections, total_height, self.a4_height, self.min_overlap
            )
            
            print(f"  Calculated {len(break_points)} A4 page breaks")
            
            screenshot_paths = []   
                                        
            for i, bp in enumerate(break_points):
                # Create filename
                if len(break_points) == 1:
                    chunk_suffix = "page_01_full"
                else:
                    chunk_suffix = f"page_{bp['chunk_index']:02d}_{bp['type']}"
                
                filename = f"{self._sanitize_filename(url, chunk_suffix)}_{timestamp}.png"
                screenshot_path = self.output_dir / "screenshots" / filename
                
                # Scroll to position with smooth scrolling
                await page.evaluate(f"""`
                    window.scrollTo({{
                        top: {bp['y']},
                        behavior: 'instant'
                    }});
                """)
                 
                # Wait for scroll and any lazy loading  
                await page.wait_for_timeout(800)
                
                # Additional wait for images and dynamic content
                try:
                    await page.wait_for_load_state('networkidle', timeout=3000)
                except:
                    pass
                
                # Take screenshot with A4 dimensions
                clip_config = {
                    'x': 0,
                    'y': 0,
                    'width': self.a4_width,
                    'height': min(bp['height'], self.a4_height)
                }
                
                await page.screenshot(
                    path=str(screenshot_path),
                    type='png',
                    clip=clip_config,
                    timeout=30000
                )
                
                screenshot_paths.append(str(screenshot_path))
                
                # Log capture info
                info_parts = [
                    f"y:{bp['y']}-{bp['y'] + bp['height']}",
                    f"h:{bp['height']}px"
                ]
                
                if 'break_reason' in bp:
                    info_parts.append(f"({bp['break_reason']})")
                
                if 'elements_preserved' in bp and bp['elements_preserved'] > 0:
                    info_parts.append(f"preserved:{bp['elements_preserved']}")
                
                print(f"  📄 Captured A4 page {bp.get('chunk_index', i+1)}: {screenshot_path.name}")
                print(f"     {' | '.join(info_parts)}")
            
            return screenshot_paths
            
        except Exception as e:
            print(f"  Error capturing visual content: {e}")
            # Fallback to single A4 screenshot
            try:
                filename = f"{self._sanitize_filename(url, 'fallback_a4')}_{timestamp}.png"
                screenshot_path = self.output_dir / "screenshots" / filename
                
                await page.screenshot(
                    path=str(screenshot_path),
                    full_page=False,
                    type='png',
                    timeout=15000,
                    clip={
                        'x': 0,
                        'y': 0,
                        'width': self.a4_width,
                        'height': self.a4_height
                    }
                )
                return [str(screenshot_path)]
            except Exception as fallback_error:
                print(f"  Fallback A4 screenshot also failed: {fallback_error}")
                return []

    async def _extract_text_content(self, page: Page, url: str, timestamp: int) -> Dict[str, Any]:
        """Extract and process text content from the page with preserved formatting
        
        Args:
            page: Playwright Page instance
            url: Page URL for naming
            timestamp: Timestamp for unique filenames
            
        Returns:
            Dictionary containing:
            - text_content: Formatted text with preserved structure
            - html_content: Raw HTML (if enabled)
            - links: List of URLs
            - metadata: Extracted metadata
            
        Notes:
            - Preserves HTML formatting as markdown-like text
            - Uses configured content selectors
            - Saves text/HTML to files
            - Extracts links/images/metadata
        """
        try:
            # NEW: Content element detection logic
            content_element = None
            using_fallback = False
            temp_id = None
            
            # Try configured selectors first
            for selector in self.content_selectors:
                content_element = await page.query_selector(selector)
                if content_element:
                    content_text = await content_element.inner_text()
                    if len(content_text) >= self.min_content_length:
                        break
                    content_element = None
            
            # Fallback to automatic detection if no suitable element found
            if not content_element:
                print("  Using fallback content detection")
                using_fallback = True
                optimal_html = await self._find_optimal_content_element(page)
                if optimal_html:
                    # Create a temporary element from the detected HTML
                    temp_id = "__content_fallback__"
                    await page.evaluate(f"""(temp_id, optimal_html) => {{
                        const temp = document.createElement('div');
                        temp.id = temp_id;
                        temp.innerHTML = optimal_html;
                        document.body.appendChild(temp);
                    }}""", temp_id, optimal_html)
                    content_element = await page.query_selector(f'#{temp_id}')
            
            # If still no element, use body as final fallback
            if not content_element:
                content_element = await page.query_selector('body')
            
            # Get page content using multiple strategies with formatting preservation
            content_data = await page.evaluate("""(params) => {
                const result = {
                    title: document.title || '',
                    text: '',
                    html: '',
                    links: [],
                    images: [],
                    metadata: {}
                };
                
                // Get the content element
                let contentElement = document.body;
                if (params.contentElementId) {
                    contentElement = document.getElementById(params.contentElementId);
                }
                
                if (!contentElement) {
                    contentElement = document.body;
                }
                
                // Function to convert HTML to formatted text while preserving structure
                function htmlToFormattedText(element) {
                    let text = '';
                    
                    function processNode(node, depth = 0) {
                        const indent = '  '.repeat(depth);
                        
                        if (node.nodeType === Node.TEXT_NODE) {
                            const textContent = node.textContent.trim();
                            if (textContent) {
                                text += textContent + ' ';
                            }
                            return;
                        }
                        
                        if (node.nodeType !== Node.ELEMENT_NODE) {
                            return;
                        }
                        
                        const tagName = node.tagName.toLowerCase();
                        
                        // Handle different HTML elements with appropriate formatting
                        switch (tagName) {
                            case 'h1':
                                text += '\\n\\n# ';
                                break;
                            case 'h2':
                                text += '\\n\\n## ';
                                break;
                            case 'h3':
                                text += '\\n\\n### ';
                                break;
                            case 'h4':
                                text += '\\n\\n#### ';
                                break;
                            case 'h5':
                                text += '\\n\\n##### ';
                                break;
                            case 'h6':
                                text += '\\n\\n###### ';
                                break;
                            case 'p':
                                text += '\\n\\n';
                                break;
                            case 'br':
                                text += '\\n';
                                return; // br is self-closing
                            case 'hr':
                                text += '\\n\\n---\\n\\n';
                                return; // hr is self-closing
                            case 'div':
                            case 'section':
                            case 'article':
                                text += '\\n';
                                break;
                            case 'blockquote':
                                text += '\\n\\n> ';
                                break;
                            case 'pre':
                                text += '\\n\\n```\\n';
                                break;
                            case 'code':
                                if (node.parentElement && node.parentElement.tagName.toLowerCase() !== 'pre') {
                                    text += '`';
                                }
                                break;
                            case 'strong':
                            case 'b':
                                text += '**';
                                break;
                            case 'em':
                            case 'i':
                                text += '*';
                                break;
                            case 'u':
                                text += '_';
                                break;
                            case 'del':
                            case 's':
                                text += '~~';
                                break;
                            case 'ul':
                                text += '\\n';
                                break;
                            case 'ol':
                                text += '\\n';
                                break;
                            case 'li':
                                const listParent = node.closest('ol, ul');
                                if (listParent && listParent.tagName.toLowerCase() === 'ol') {
                                    const index = Array.from(listParent.children).indexOf(node) + 1;
                                    text += `\\n${indent}${index}. `;
                                } else {
                                    text += `\\n${indent}- `;
                                }
                                break;
                            case 'table':
                                text += '\\n\\n';
                                break;
                            case 'tr':
                                text += '\\n';
                                break;
                            case 'td':
                            case 'th':
                                text += ' | ';
                                break;
                            case 'thead':
                                text += '\\n';
                                break;
                            case 'tbody':
                                text += '\\n';
                                break;
                            case 'a':
                                const href = node.getAttribute('href');
                                if (href) {
                                    text += '[';
                                }
                                break;
                            case 'img':
                                const alt = node.getAttribute('alt') || '';
                                const src = node.getAttribute('src') || '';
                                text += `![${alt}](${src})`;
                                return; // img is self-closing
                            case 'sup':
                                text += '^';
                                break;
                            case 'sub':
                                text += '_';
                                break;
                            case 'span':
                                // Check for special classes that might indicate formatting
                                const className = node.className || '';
                                if (className.includes('bold') || className.includes('strong')) {
                                    text += '**';
                                } else if (className.includes('italic') || className.includes('emphasis')) {
                                    text += '*';
                                }
                                break;
                        }
                        
                        // Process child nodes
                        for (const child of node.childNodes) {
                            processNode(child, depth + 1);
                        }
                        
                        // Handle closing tags
                        switch (tagName) {
                            case 'h1':
                            case 'h2':
                            case 'h3':
                            case 'h4':
                            case 'h5':
                            case 'h6':
                                text += '\\n';
                                break;
                            case 'pre':
                                text += '\\n```\\n';
                                break;
                            case 'code':
                                if (node.parentElement && node.parentElement.tagName.toLowerCase() !== 'pre') {
                                    text += '`';
                                }
                                break;
                            case 'strong':
                            case 'b':
                                text += '**';
                                break;
                            case 'em':
                            case 'i':
                                text += '*';
                                break;
                            case 'u':
                                text += '_';
                                break;
                            case 'del':
                            case 's':
                                text += '~~';
                                break;
                            case 'a':
                                const href = node.getAttribute('href');
                                if (href) {
                                    text += `](${href})`;
                                }
                                break;
                            case 'sup':
                            case 'sub':
                                text += ' ';
                                break;
                            case 'span':
                                const className = node.className || '';
                                if (className.includes('bold') || className.includes('strong')) {
                                    text += '**';
                                } else if (className.includes('italic') || className.includes('emphasis')) {
                                    text += '*';
                                }
                                break;
                        }
                    }
                    
                    processNode(element);
                    return text;
                }
                
                // Extract formatted text content
                result.text = htmlToFormattedText(contentElement);
                
                // Extract HTML if needed
                if (params.saveHtml) {
                    result.html = contentElement.innerHTML || '';
                }
                
                // Extract links
                if (params.extractLinks) {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    result.links = links.map(link => {
                        return {
                            url: link.href,
                            text: link.textContent.trim(),
                            title: link.title || '',
                            href_attribute: link.getAttribute('href') // Keep original href for filtering
                        };
                    }).filter(link => {
                        // Exclude anchor links (starting with #)
                        const originalHref = link.href_attribute;
                        return link.url.startsWith('http') && 
                            !originalHref.startsWith('#') &&
                            !originalHref.startsWith('javascript:') &&
                            !originalHref.startsWith('mailto:') &&
                            !originalHref.startsWith('tel:');
                    });
                }
                
                // Extract images
                if (params.extractImages) {
                    const images = Array.from(document.querySelectorAll('img[src]'));
                    result.images = images.map(img => {
                        return {
                            src: img.src,
                            alt: img.alt || '',
                            title: img.title || '',
                            width: img.naturalWidth || img.width || 0,
                            height: img.naturalHeight || img.height || 0
                        };
                    });
                }
                
                // Extract metadata
                const metaTags = Array.from(document.querySelectorAll('meta'));
                metaTags.forEach(meta => {
                    if (meta.name) {
                        result.metadata[meta.name] = meta.content || '';
                    } else if (meta.property) {
                        result.metadata[meta.property] = meta.content || '';
                    }
                });
                
                return result;
            }""", {
                'saveHtml': self.save_html,
                'extractLinks': self.extract_links,
                'extractImages': self.extract_images,
                'contentElementId': temp_id if using_fallback else None
            })
            
            # Clean text if requested (but preserve formatting)
            if self.clean_text and content_data['text']:
                content_data['text'] = self._clean_formatted_text(content_data['text'])
            
            # Save text content to file
            text_filename = f"{self._sanitize_filename(url, 'content')}_{timestamp}.txt"
            text_path = self.output_dir / "text" / text_filename
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n")
                f.write(f"Title: {content_data['title']}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write("=" * 50 + "\n\n")
                f.write(content_data['text'])
                
                if content_data['links']:
                    f.write("\n\n" + "=" * 50)
                    f.write("\nEXTRACTED LINKS:\n")
                    for link in content_data['links']:
                        f.write(f"- {link['text']}: {link['url']}\n")
                
                if content_data['images']:
                    f.write("\n\n" + "=" * 50)
                    f.write("\nEXTRACTED IMAGES:\n")
                    for img in content_data['images']:
                        f.write(f"- {img['alt']}: {img['src']} ({img['width']}x{img['height']})\n")
            
            # Save HTML if requested
            if self.save_html and content_data['html']:
                html_filename = f"{self._sanitize_filename(url, 'content')}_{timestamp}.html"
                html_path = self.output_dir / "html" / html_filename
                
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(content_data['html'])
            
            print(f"  📝 Extracted {len(content_data['text'])} characters of formatted text")
            print(f"  🔗 Found {len(content_data['links'])} links and {len(content_data['images'])} images")
            
            # NEW: Clean up temporary element if we used fallback
            if using_fallback and temp_id:
                await page.evaluate(f"""(temp_id) => {{
                    const el = document.getElementById(temp_id);
                    if (el) el.remove();
                }}""", temp_id)
            
            return {
                'text_content': content_data['text'],
                'html_content': content_data['html'] if self.save_html else '',
                'links': [link['url'] for link in content_data['links']],
                'metadata': {
                    'title': content_data['title'],
                    'link_details': content_data['links'],
                    'image_details': content_data['images'],
                    'meta_tags': content_data['metadata'],
                    'text_file': str(text_path),
                    'html_file': str(html_path) if self.save_html else None
                }
            }
            
        except Exception as e:
            print(f"  Error extracting text content: {e}")
            return {
                'text_content': '',
                'html_content': '',
                'links': [],
                'metadata': {'error': str(e)}
            }

    def _clean_formatted_text(self, text: str) -> str:
        """Clean formatted text while preserving structure
        
        Args:
            text: Raw extracted formatted text
            
        Returns:
            Cleaned text with preserved formatting but normalized whitespace
            
        Notes:
            - Preserves markdown-like formatting
            - Normalizes spacing between elements
            - Removes excessive empty lines
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive whitespace but preserve intentional spacing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Clean up each line but preserve indentation
            stripped = line.rstrip()
            if stripped or (cleaned_lines and not cleaned_lines[-1].strip()):
                # Keep line if it has content, or if it's an empty line after content
                cleaned_lines.append(stripped)
        
        # Rejoin lines
        text = '\n'.join(cleaned_lines)
        
        # Reduce excessive consecutive empty lines to maximum of 2
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Clean up spacing around formatting markers
        text = re.sub(r'\s+([*_`~])', r' \1', text)  # Space before formatting
        text = re.sub(r'([*_`~])\s+', r'\1 ', text)  # Space after formatting
        
        # Clean up excessive spaces
        text = re.sub(r' {3,}', '  ', text)  # Max 2 consecutive spaces
        
        # Remove control characters but preserve formatting
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        return text.strip()

    async def _process_page(self, page: Page, url: str) -> CrawlResult:
        """Process a single page based on crawling mode
        
        Args:
            page: Playwright Page instance
            url: URL to process
            
        Returns:
            CrawlResult object with capture/extraction results
            
        Notes:
            - Handles retries for failed pages
            - Combines visual/text extraction results
            - Records page dimensions and metadata"""
        
        for attempt in range(self.retry_attempts + 1):
            try:
                print(f"🌐 Processing {url} (attempt {attempt + 1}/{self.retry_attempts + 1})")
                timestamp = int(time.time())
                await page.goto(url, wait_until='load')
                await page.wait_for_timeout(self.wait_time)

                result = CrawlResult(
                    url=url,
                    timestamp=timestamp,
                    mode=self.mode,
                    status_code=200  # Playwright does not expose response code directly
                )

                if self.mode in ["visual", "both"]:
                    result.screenshot_paths = await self._capture_visual_content(page, url, timestamp)

                if self.mode in ["text", "both"]:
                    text_data = await self._extract_text_content(page, url, timestamp)
                    result.text_content = text_data.get("text_content", "")
                    result.html_content = text_data.get("html_content", "")
                    result.links = text_data.get("links", [])
                    result.metadata = text_data.get("metadata", {})

                # Track total page height if captured
                result.total_height = await page.evaluate("document.body.scrollHeight")
                result.page_count = len(result.screenshot_paths)

                return result

            except Exception as e:
                print(f"  Error on attempt {attempt + 1}: {e}")
                last_error = str(e)

        # Return error result if all retries failed
        return CrawlResult(
            url=url,
            timestamp=time.time(),
            mode=self.mode,
            error=last_error
        )

    async def crawl_recursive(self, start_urls: List[str]) -> List[CrawlResult]:
        """
        Recursively crawl websites starting from the given URLs
        
        Args:
            start_urls: List of starting URLs
            
        Returns:
            List of CrawlResult objects from all crawled pages
        """
        browser, context = await self._setup_browser()
        results = []
        
        try:
            # Initialize crawling state
            self._initialize_crawl_queue(start_urls)
            pages_crawled = 0
            
            while self.crawl_queue and pages_crawled < self.max_pages:
                # Get next URL from queue
                current_url, current_depth = self.crawl_queue.pop(0)
                
                # Skip if already visited (shouldn't happen, but safety check)
                if current_url in self.visited_urls:
                    continue
                
                # Mark as visited
                self.visited_urls.add(current_url)
                pages_crawled += 1
                
                print(f"\n[Depth {current_depth}] Crawling {current_url} ({pages_crawled}/{self.max_pages})")
                
                # Create new page and process
                page = await context.new_page()
                try:
                    result = await self._process_page(page, current_url)
                    result.metadata = result.metadata or {}
                    result.metadata['crawl_depth'] = current_depth
                    
                    self.crawl_results.append(result)
                    results.append(result)
                    
                    # Extract links for further crawling (only if we haven't reached max depth)
                    if current_depth < self.max_depth - 1:  # -1 because we want to crawl AT max_depth, not beyond
                        new_links = self._extract_links_from_result(result, current_depth)
                        
                        # Add new links to queue
                        for link in new_links:
                            if link not in self.visited_urls and link not in [url for url, _ in self.crawl_queue]:
                                self.crawl_queue.append((link, current_depth + 1))
                                self.url_to_depth[link] = current_depth + 1
                        
                        print(f"  Added {len(new_links)} new URLs to crawl queue (total queue: {len(self.crawl_queue)})")
                    
                    # Respectful delay between requests
                    if self.delay_between_requests > 0:
                        await asyncio.sleep(self.delay_between_requests)
                        
                except Exception as e:
                    print(f"  Error processing {current_url}: {e}")
                    # Add error result
                    error_result = CrawlResult(
                        url=current_url,
                        timestamp=time.time(),
                        mode=self.mode,
                        error=str(e),
                        metadata={'crawl_depth': current_depth}
                    )
                    results.append(error_result)
                    
                finally:
                    await page.close()
            
            print(f"\nRecursive crawl completed:")
            print(f"  Total pages crawled: {pages_crawled}")
            print(f"  Total URLs visited: {len(self.visited_urls)}")
            print(f"  Remaining queue: {len(self.crawl_queue)}")
            
            # Print depth statistics
            depth_stats = {}
            for result in results:
                depth = result.metadata.get('crawl_depth', 0) if result.metadata else 0
                depth_stats[depth] = depth_stats.get(depth, 0) + 1
            
            print(f"  Pages by depth: {depth_stats}")
            
        finally:
            await self._cleanup_browser(browser)
        
        return results

    async def _crawl_single_level(self, urls: List[str]) -> List[CrawlResult]:
        """
        Original single-level crawling method (renamed from crawl)
        
        Args:
            urls: List of URLs to crawl
            
        Returns:
            List of CrawlResult objects
        """
        browser, context = await self._setup_browser()
        results = []

        try:
            for i, url in enumerate(urls):
                if i >= self.max_pages:
                    break
                if url in self.visited_urls:
                    continue
                self.visited_urls.add(url)

                page = await context.new_page()
                try:
                    result = await self._process_page(page, url)
                    result.metadata = result.metadata or {}
                    result.metadata['crawl_depth'] = 0
                    self.crawl_results.append(result)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {url}: {e}")
                finally:
                    await page.close()

        finally:
            await self._cleanup_browser(browser)

        return results

    def _save_cleaned_text_files(self, results: List[CrawlResult]):
        """
        Save cleaned text content to .txt files after boilerplate removal
        """
        for result in results:
            if result.text_content and result.metadata:
                path = result.metadata.get('text_file')
                if path:
                    try:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(f"URL: {result.url}\n")
                            f.write(f"Title: {result.title}\n")
                            f.write(f"Timestamp: {result.timestamp}\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(result.text_content)
                    except Exception as e:
                        print(f"Failed to save cleaned text to {path}: {e}")

    def _remove_boilerplate_text(self, results: List[CrawlResult]) -> None:
        """
        Remove common boilerplate text (headers, footers, sidebars) from crawl results
        
        Args:
            results: List of CrawlResult objects
            
        Notes:
            - Groups pages by domain
            - Uses shingling to identify common text blocks
            - Removes lines that appear in boilerplate shingles
        """
        # Group results by domain
        domain_groups = defaultdict(list)
        for result in results:
            if result.error or not result.text_content:
                continue
            parsed = urlparse(result.url)
            domain = parsed.netloc
            domain_groups[domain].append(result)

        # Process each domain group
        for domain, domain_results in domain_groups.items():
            num_pages = len(domain_results)
            if num_pages < 2:
                continue  # Need at least 2 pages to find common patterns

            print(f"\n🧹 Removing boilerplate for domain: {domain} ({num_pages} pages)")
            
            # Calculate threshold for boilerplate detection
            threshold = max(2, int(num_pages * self.boilerplate_threshold))
            shingle_size = self.boilerplate_shingle_size
            
            # Count shingle frequencies
            shingle_freq = defaultdict(int)
            for result in domain_results:
                lines = result.text_content.splitlines()
                for i in range(len(lines) - shingle_size + 1):
                    shingle = tuple(lines[i:i + shingle_size])
                    shingle_freq[shingle] += 1

            # Identify boilerplate shingles
            boilerplate_set = set()
            for shingle, count in shingle_freq.items():
                if count >= threshold:
                    boilerplate_set.add(shingle)
            
            if not boilerplate_set:
                print(f"  No boilerplate found for {domain}")
                continue
            
            print(f"  Found {len(boilerplate_set)} boilerplate patterns")
            
            # Remove boilerplate from each page
            for result in domain_results:
                lines = result.text_content.splitlines()
                is_boilerplate = [False] * len(lines)
                
                # Mark lines that are part of boilerplate shingles
                for i in range(len(lines) - shingle_size + 1):
                    shingle = tuple(lines[i:i + shingle_size])
                    if shingle in boilerplate_set:
                        for j in range(i, i + shingle_size):
                            is_boilerplate[j] = True
                
                # Filter out boilerplate lines
                cleaned_lines = [
                    line for i, line in enumerate(lines)
                    if not is_boilerplate[i]
                ]
                
                # Calculate stats
                original_line_count = len(lines)
                removed_line_count = sum(is_boilerplate)
                removal_percentage = (removed_line_count / original_line_count) * 100
                
                # Update result
                result.text_content = "\n".join(cleaned_lines)
                result.metadata["boilerplate_removed"] = {
                    "original_lines": original_line_count,
                    "removed_lines": removed_line_count,
                    "removal_percentage": round(removal_percentage, 1),
                    "shingle_size": shingle_size,
                    "threshold": threshold
                }
                
                print(f"  Removed {removed_line_count}/{original_line_count} lines "
                      f"({removal_percentage:.1f}%) from {result.url}")

    async def crawl(self, urls: List[str]) -> List[CrawlResult]:
        """
        Main crawl method - now supports both single-level and recursive crawling
        
        Args:
            urls: List of URLs to crawl
            
        Returns:
            List of CrawlResult objects
        """
        if self.max_depth <= 1:
            # Use original single-level crawling
            results = await self._crawl_single_level(urls)
        else:
            # Use new recursive crawling
            results = await self.crawl_recursive(urls)
        
        # Post-process text content to remove boilerplate
        if self.mode in ["text", "both"] and results:
            self._remove_boilerplate_text(results)
            self._save_cleaned_text_files(results)
        
        return results

# Updated main function to demonstrate recursive crawling
async def main():
    # Sample URLs for testing recursive crawling
    test_urls = [
        "https://www.reva.edu.in/vice-chancellor"
    ]

    # Initialize crawler with recursive settings
    crawler = EnhancedCrawler(
        mode="text",
        output_dir="crawler_output_recursive",
        page_width=1920,
        smart_splitting=True,
        preserve_context=True,
        headless=True,
        max_pages=3,  # Limit total pages
        wait_time=2000,
        save_html=False,
        clean_text=True,
        # Recursive crawling settings
        max_depth=3,  # Crawl 3 levels deep
        same_domain_only=True,
        url_patterns=[
            r'.*reva\.edu\.in/.*',  # Only Reva articles
        ],
        exclude_patterns=[
            r'.*\.(jpg|jpeg|png|gif|pdf|doc)$',  # Skip media files
            r'.*[Cc]ategory:.*',  # Skip Wikipedia categories
            r'.*[Tt]alk:.*',  # Skip talk pages
            r'.*[Uu]ser:.*',  # Skip user pages
        ],
        delay_between_requests=1.0,  # Be respectful with 1 second delays
        min_content_length=300,  # Minimum content length threshold
        # Boilerplate removal settings
        boilerplate_shingle_size=3,
        boilerplate_threshold=0.5
    )

    # Run crawl
    results = await crawler.crawl(test_urls)

    # Print results summary
    print("\n" + "="*60)
    print("CRAWL RESULTS SUMMARY")
    print("="*60)
    
    for res in results:
        depth = res.metadata.get('crawl_depth', 0) if res.metadata else 0
        status = '✅ Success' if not res.error else '❌ Error'
        indent = "  " * depth
        
        print(f"{indent}[Depth {depth}] {res.url}")
        print(f"{indent}Status: {status}")
        if res.error:
            print(f"{indent}Error: {res.error}")
        else:
            print(f"{indent}Text Length: {len(res.text_content)} chars")
            print(f"{indent}Links Found: {len(res.links)}")
            # Show boilerplate removal stats if available
            if "boilerplate_removed" in res.metadata:
                bp = res.metadata["boilerplate_removed"]
                print(f"{indent}Boilerplate Removed: {bp['removed_lines']} lines "
                      f"({bp['removal_percentage']}%)")
        print()


if __name__ == "__main__":
    import asyncio
    import time
    asyncio.run(main())