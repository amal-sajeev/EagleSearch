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

from playwright.async_api import async_playwright, Page, Browser, BrowserContext


@dataclass
class CrawlResult:
    """Data class to store crawl results"""
    url: str
    screenshot_paths: List[str] = None  # For visual mode
    text_content: str = ""  # For text mode
    html_content: str = ""  # Raw HTML if needed
    status_code: int = 0
    title: str = ""
    timestamp: float = 0
    total_height: int = 0
    page_count: int = 0
    links: List[str] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    mode: str = "visual"  # "visual", "text", or "both"

    def __post_init__(self):
        if self.screenshot_paths is None:
            self.screenshot_paths = []
        if self.links is None:
            self.links = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class A4Dimensions:
    """A4 page dimensions in pixels at different DPIs"""
    # A4 size: 210 × 297 mm
    WIDTH_72DPI = 595   # 72 DPI (web standard)
    HEIGHT_72DPI = 842
    WIDTH_96DPI = 794   # 96 DPI (Windows standard)
    HEIGHT_96DPI = 1123
    WIDTH_150DPI = 1240 # 150 DPI (high quality)
    HEIGHT_150DPI = 1754
    WIDTH_DESKTOP = 1920
    HEIGHT_DESKTOP = 1080
    
    @classmethod
    def get_dimensions(cls, dpi: int = 96) -> Tuple[int, int]:
        """Get A4 dimensions for specified DPI"""
        if dpi <= 96:
            return cls.WIDTH_96DPI, cls.HEIGHT_96DPI
        elif dpi>96 and dpi<=150:
            return cls.WIDTH_150DPI, cls.HEIGHT_150DPI
        else:
            return cls.HEIGHT_DESKTOP, cls.WIDTH_DESKTOP


class ContentAnalyzer:
    """Analyzes page content to determine intelligent break points"""
    
    @staticmethod
    async def get_content_sections(page: Page) -> List[Dict[str, Any]]:
        """Get semantic content sections with priorities"""
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
        """Find safe break points that don't cut important content"""
        
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
    An enhanced website crawler that supports both visual and text crawling modes
    with intelligent A4-sized page splitting
    """
    
    def __init__(
        self,
        mode: str = "visual",  # "visual", "text", or "both"
        output_dir: str = "crawler_output",
        # A4 visual settings
        a4_dpi: int = 96,  # DPI for A4 sizing (72, 96, or 150)
        min_overlap: int = 50,  # Minimum overlap between pages
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
        content_selectors: List[str] = None
    ):
        """
        Initialize the enhanced crawler
        
        Args:
            mode: Crawling mode - "visual", "text", or "both"
            output_dir: Directory to save outputs
            a4_dpi: DPI setting for A4 page dimensions (72, 96, or 150)
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
        """
        self.mode = mode.lower()
        if self.mode not in ["visual", "text", "both"]:
            raise ValueError("Mode must be 'visual', 'text', or 'both'")
            
        self.output_dir = Path(output_dir)
        
        # A4 dimensions
        self.a4_width, self.a4_height = A4Dimensions.get_dimensions(a4_dpi)
        self.a4_dpi = a4_dpi
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
        
        # Track visited URLs and results
        self.visited_urls = set()
        self.crawl_results = []
        
        print(f"Initialized crawler with A4 dimensions: {self.a4_width}x{self.a4_height}px at {a4_dpi}DPI")
    
    async def _setup_browser(self):
        """Setup and configure the browser with A4 viewport"""
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

    async def _capture_visual_content(self, page: Page, url: str, timestamp: int) -> List[str]:
        """Capture visual content with intelligent A4-sized splitting"""
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
                await page.evaluate(f"""
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
        """Extract and process text content from the page"""
        try:
            # Get page content using multiple strategies
            content_data = await page.evaluate(f"""
                () => {{
                    const result = {{
                        title: document.title || '',
                        text: '',
                        html: '',
                        links: [],
                        images: [],
                        metadata: {{}}
                    }};
                    
                    // Try to get main content using selectors
                    const selectors = {json.dumps(self.content_selectors)};
                    let contentElement = null;
                    
                    for (const selector of selectors) {{
                        contentElement = document.querySelector(selector);
                        if (contentElement && contentElement.textContent.trim().length > 100) {{
                            break;
                        }}
                    }}
                    
                    if (!contentElement) {{
                        contentElement = document.body;
                    }}
                    
                    // Extract text content
                    result.text = contentElement.textContent || contentElement.innerText || '';
                    
                    // Extract HTML if needed
                    if ({json.dumps(self.save_html)}) {{
                        result.html = contentElement.innerHTML || '';
                    }}
                    
                    // Extract links
                    if ({json.dumps(self.extract_links)}) {{
                        const links = Array.from(document.querySelectorAll('a[href]'));
                        result.links = links.map(link => {{
                            return {{
                                url: link.href,
                                text: link.textContent.trim(),
                                title: link.title || ''
                            }};
                        }}).filter(link => link.url.startsWith('http'));
                    }}
                    
                    // Extract images
                    if ({json.dumps(self.extract_images)}) {{
                        const images = Array.from(document.querySelectorAll('img[src]'));
                        result.images = images.map(img => {{
                            return {{
                                src: img.src,
                                alt: img.alt || '',
                                title: img.title || '',
                                width: img.naturalWidth || img.width || 0,
                                height: img.naturalHeight || img.height || 0
                            }};
                        }});
                    }}
                    
                    // Extract metadata
                    const metaTags = Array.from(document.querySelectorAll('meta'));
                    metaTags.forEach(meta => {{
                        if (meta.name) {{
                            result.metadata[meta.name] = meta.content || '';
                        }} else if (meta.property) {{
                            result.metadata[meta.property] = meta.content || '';
                        }}
                    }});
                    
                    return result;
                }}
            """)
            
            # Clean text if requested
            if self.clean_text and content_data['text']:
                content_data['text'] = self._clean_text(content_data['text'])
            
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
            
            print(f"  📝 Extracted {len(content_data['text'])} characters of text")
            print(f"  🔗 Found {len(content_data['links'])} links and {len(content_data['images'])} images")
            
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
    
    def _clean_text(self, text: str) -> str:
        """Clean and format extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Clean up common artifacts
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Remove repeated special characters
        text = re.sub(r'([^\w\s])\1{3,}', r'\1\1', text)
        
        return text.strip()

    async def _process_page(self, page: Page, url: str) -> CrawlResult:
        """Process a single page based on the crawling mode"""
        
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

    async def crawl(self, urls: List[str]) -> List[CrawlResult]:
        """Crawl the given list of URLs"""
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
                    self.crawl_results.append(result)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {url}: {e}")
                finally:
                    await page.close()

        finally:
            await self._cleanup_browser(browser)

        return results

async def main():
    # Sample URLs for testing (replace or expand as needed)
    test_urls = [
        "https://www.reva.edu.in"
    ]

    # Initialize crawler (visual + text mode)
    crawler = EnhancedCrawler(
        mode="both",
        output_dir="crawler_output_test",
        a4_dpi=151,
        smart_splitting=True,
        preserve_context=True,
        headless=True,
        max_pages=2,
        wait_time=3000,
        save_html=True
    )

    results = await crawler.crawl(test_urls)

    # Print results summary
    for res in results:
        print("\n========== RESULT ==========")
        print(f"URL: {res.url}")
        print(f"Status: {'✅ Success' if not res.error else '❌ Error'}")
        print(f"Title: {res.title}")
        print(f"Total Height: {res.total_height}px")
        print(f"Page Count: {res.page_count}")
        print(f"Screenshots: {res.screenshot_paths}")
        print(f"Text Length: {len(res.text_content)}")
        print(f"Links Extracted: {len(res.links)}")
        if res.error:
            print(f"Error: {res.error}")
        print("=============================")

if __name__ == "__main__":
    asyncio.run(main())
