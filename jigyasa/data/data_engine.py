"""
Autonomous Data Acquisition Engine
Handles web scraping, search, and data collection for continuous learning
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None
import time
import random
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import re
from urllib.parse import urljoin, urlparse
import logging
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from datetime import datetime, timedelta

from ..config import DataConfig


@dataclass
class SearchResult:
    """Container for search results"""
    title: str
    url: str
    snippet: str
    domain: str
    relevance_score: float
    metadata: Dict[str, Any]


@dataclass
class ScrapedContent:
    """Container for scraped web content"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    content_hash: str
    quality_score: float


class SearchEngine:
    """
    Search engine interface for generating queries and retrieving results
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.search_providers = {
            'duckduckgo': self._search_duckduckgo,
            'bing': self._search_bing,
            'google': self._search_google_custom
        }
        
        # Query templates for different domains
        self.query_templates = {
            'scientific': [
                "{topic} research papers",
                "{topic} scientific study",
                "{topic} academic articles",
                "recent advances {topic}",
                "{topic} methodology"
            ],
            'technical': [
                "{topic} tutorial",
                "{topic} documentation",
                "{topic} implementation guide",
                "how to {topic}",
                "{topic} best practices"
            ],
            'factual': [
                "{topic} facts",
                "{topic} information",
                "what is {topic}",
                "{topic} overview",
                "{topic} explanation"
            ],
            'recent': [
                "{topic} 2024",
                "{topic} latest news",
                "recent {topic}",
                "{topic} updates",
                "new {topic}"
            ]
        }
    
    def generate_search_queries(
        self,
        topic: str,
        domain: str = 'general',
        num_queries: int = 5,
        include_variations: bool = True
    ) -> List[str]:
        """
        Generate diverse search queries for a given topic
        
        Args:
            topic: Main topic to search for
            domain: Domain type (scientific, technical, factual, recent)
            num_queries: Number of queries to generate
            include_variations: Whether to include query variations
        """
        queries = []
        
        # Get templates for the domain
        templates = self.query_templates.get(domain, self.query_templates['factual'])
        
        # Generate base queries
        for template in templates[:num_queries]:
            query = template.format(topic=topic)
            queries.append(query)
        
        # Add variations if requested
        if include_variations and len(queries) < num_queries:
            variations = self._generate_query_variations(topic)
            queries.extend(variations[:num_queries - len(queries)])
        
        return queries[:num_queries]
    
    def _generate_query_variations(self, topic: str) -> List[str]:
        """Generate variations of the topic for broader search"""
        variations = []
        
        # Add synonyms and related terms
        if ' ' in topic:
            words = topic.split()
            # Try different word orders
            variations.append(' '.join(reversed(words)))
            # Try partial phrases
            if len(words) > 2:
                variations.append(' '.join(words[:-1]))
                variations.append(' '.join(words[1:]))
        
        # Add question formats
        variations.extend([
            f"what is {topic}",
            f"how does {topic} work",
            f"why {topic}",
            f"{topic} explained"
        ])
        
        return variations
    
    async def search(
        self,
        query: str,
        provider: str = 'duckduckgo',
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Perform search using specified provider
        
        Args:
            query: Search query
            provider: Search provider to use
            max_results: Maximum number of results to return
        """
        search_fn = self.search_providers.get(provider, self._search_duckduckgo)
        
        try:
            results = await search_fn(query, max_results)
            return results
        except Exception as e:
            logging.error(f"Search failed for query '{query}': {str(e)}")
            return []
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    results = []
                    
                    # Process instant answer
                    if data.get('AbstractText'):
                        results.append(SearchResult(
                            title=data.get('AbstractSource', 'DuckDuckGo'),
                            url=data.get('AbstractURL', ''),
                            snippet=data.get('AbstractText', ''),
                            domain=urlparse(data.get('AbstractURL', '')).netloc,
                            relevance_score=0.9,
                            metadata={'source': 'instant_answer'}
                        ))
                    
                    # Process related topics
                    for topic in data.get('RelatedTopics', [])[:max_results]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            results.append(SearchResult(
                                title=topic.get('Text', '').split(' - ')[0],
                                url=topic.get('FirstURL', ''),
                                snippet=topic.get('Text', ''),
                                domain=urlparse(topic.get('FirstURL', '')).netloc,
                                relevance_score=0.7,
                                metadata={'source': 'related_topic'}
                            ))
                    
                    return results[:max_results]
            
            except Exception as e:
                logging.error(f"DuckDuckGo search error: {str(e)}")
                return []
    
    async def _search_bing(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Bing (requires API key)"""
        # This would require a Bing Search API key
        # For now, return empty results
        logging.warning("Bing search not implemented - requires API key")
        return []
    
    async def _search_google_custom(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Google Custom Search (requires API key)"""
        # This would require a Google Custom Search API key
        # For now, return empty results
        logging.warning("Google Custom Search not implemented - requires API key")
        return []


class WebScraper:
    """
    Web scraper for extracting content from URLs
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.session = None
        self.browser = None
        
        # Content extractors for different site types
        self.extractors = {
            'article': self._extract_article_content,
            'wiki': self._extract_wiki_content,
            'academic': self._extract_academic_content,
            'generic': self._extract_generic_content
        }
        
        # Site-specific configurations
        self.site_configs = {
            'wikipedia.org': {'type': 'wiki', 'wait_time': 1},
            'arxiv.org': {'type': 'academic', 'wait_time': 2},
            'scholar.google.com': {'type': 'academic', 'wait_time': 3},
            'medium.com': {'type': 'article', 'wait_time': 1},
            'default': {'type': 'generic', 'wait_time': 1}
        }
    
    async def scrape_url(self, url: str, use_js: bool = False) -> Optional[ScrapedContent]:
        """
        Scrape content from a single URL
        
        Args:
            url: URL to scrape
            use_js: Whether to use JavaScript rendering (Playwright)
        """
        try:
            if use_js:
                content = await self._scrape_with_playwright(url)
            else:
                content = await self._scrape_with_requests(url)
            
            if content:
                # Calculate quality score
                quality_score = self._calculate_quality_score(content)
                
                # Create content hash for deduplication
                content_hash = hashlib.md5(content['content'].encode()).hexdigest()
                
                return ScrapedContent(
                    url=url,
                    title=content['title'],
                    content=content['content'],
                    metadata=content['metadata'],
                    timestamp=datetime.now(),
                    content_hash=content_hash,
                    quality_score=quality_score
                )
            
        except Exception as e:
            logging.error(f"Failed to scrape {url}: {str(e)}")
        
        return None
    
    async def _scrape_with_requests(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape using requests and BeautifulSoup"""
        headers = {
            'User-Agent': self.config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Determine site type
                        domain = urlparse(url).netloc
                        site_config = self.site_configs.get(domain, self.site_configs['default'])
                        
                        # Extract content using appropriate method
                        extractor = self.extractors[site_config['type']]
                        content = extractor(soup, url)
                        
                        # Add response metadata
                        content['metadata'].update({
                            'status_code': response.status,
                            'content_type': response.headers.get('content-type', ''),
                            'content_length': len(html),
                            'domain': domain
                        })
                        
                        return content
            
            except Exception as e:
                logging.error(f"Requests scraping error for {url}: {str(e)}")
        
        return None
    
    async def _scrape_with_playwright(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape using Playwright for JavaScript-heavy sites"""
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set user agent
                await page.set_extra_http_headers({
                    'User-Agent': self.config.user_agent
                })
                
                # Navigate to page
                await page.goto(url, wait_until='networkidle')
                
                # Wait for content to load
                await page.wait_for_timeout(2000)
                
                # Get page content
                html = await page.content()
                title = await page.title()
                
                await browser.close()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Determine site type and extract
                domain = urlparse(url).netloc
                site_config = self.site_configs.get(domain, self.site_configs['default'])
                extractor = self.extractors[site_config['type']]
                
                content = extractor(soup, url)
                content['title'] = title
                content['metadata']['rendered_with_js'] = True
                
                return content
            
            except Exception as e:
                logging.error(f"Playwright scraping error for {url}: {str(e)}")
        
        return None
    
    def _extract_article_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract content from article-style pages"""
        content = {
            'title': '',
            'content': '',
            'metadata': {'extraction_type': 'article'}
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            content['title'] = title_tag.get_text().strip()
        
        # Try multiple selectors for article content
        article_selectors = [
            'article',
            '[role="main"]',
            '.content',
            '.article-content',
            '.post-content',
            '.entry-content'
        ]
        
        article_content = None
        for selector in article_selectors:
            article_content = soup.select_one(selector)
            if article_content:
                break
        
        if not article_content:
            # Fallback to body
            article_content = soup.find('body')
        
        if article_content:
            # Remove unwanted elements
            for element in article_content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Extract text
            content['content'] = article_content.get_text(separator='\n', strip=True)
        
        return content
    
    def _extract_wiki_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract content from Wikipedia-style pages"""
        content = {
            'title': '',
            'content': '',
            'metadata': {'extraction_type': 'wiki'}
        }
        
        # Wikipedia specific selectors
        title_tag = soup.find('h1', class_='firstHeading')
        if title_tag:
            content['title'] = title_tag.get_text().strip()
        
        # Main content
        main_content = soup.find('div', id='mw-content-text')
        if main_content:
            # Remove unwanted elements
            for element in main_content.find_all(['table', 'div.navbox', 'div.infobox']):
                element.decompose()
            
            content['content'] = main_content.get_text(separator='\n', strip=True)
        
        return content
    
    def _extract_academic_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract content from academic paper pages"""
        content = {
            'title': '',
            'content': '',
            'metadata': {'extraction_type': 'academic'}
        }
        
        # Try academic-specific selectors
        title_selectors = [
            'h1.title',
            '.paper-title',
            'h1',
            'title'
        ]
        
        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element:
                content['title'] = title_element.get_text().strip()
                break
        
        # Extract abstract and content
        abstract = soup.find('div', class_='abstract')
        if abstract:
            content['content'] += "Abstract: " + abstract.get_text(strip=True) + "\n\n"
        
        # Main content
        main_selectors = [
            '.paper-content',
            '.article-content',
            'main',
            'body'
        ]
        
        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                content['content'] += main_element.get_text(separator='\n', strip=True)
                break
        
        return content
    
    def _extract_generic_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Generic content extraction"""
        content = {
            'title': '',
            'content': '',
            'metadata': {'extraction_type': 'generic'}
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            content['title'] = title_tag.get_text().strip()
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Extract main content
        main_content = soup.find('main') or soup.find('body')
        if main_content:
            content['content'] = main_content.get_text(separator='\n', strip=True)
        
        return content
    
    def _calculate_quality_score(self, content: Dict[str, Any]) -> float:
        """Calculate quality score for scraped content"""
        score = 0.0
        text = content['content']
        
        # Length score (optimal range: 500-5000 chars)
        length = len(text)
        if 500 <= length <= 5000:
            score += 0.3
        elif 200 <= length < 500 or 5000 < length <= 10000:
            score += 0.2
        elif 100 <= length < 200:
            score += 0.1
        
        # Structure score (paragraphs, sentences)
        paragraphs = text.count('\n\n')
        sentences = text.count('.')
        if paragraphs >= 2:
            score += 0.2
        if sentences >= 5:
            score += 0.2
        
        # Language quality (basic checks)
        words = text.split()
        if len(words) > 50:
            score += 0.1
            
            # Check for common English patterns
            common_words = ['the', 'and', 'a', 'an', 'is', 'are', 'was', 'were']
            common_count = sum(1 for word in words[:100] if word.lower() in common_words)
            if common_count > 5:
                score += 0.1
        
        # Title quality
        if content['title'] and len(content['title']) > 10:
            score += 0.1
        
        return min(score, 1.0)


class DataEngine:
    """
    Main data acquisition engine that coordinates search and scraping
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.search_engine = SearchEngine(config)
        self.web_scraper = WebScraper(config)
        
        # Data storage
        self.db_path = Path(config.data_cache_dir) / "data_cache.db"
        self._init_database()
        
        # Rate limiting
        self.last_request_time = {}
        
    def _init_database(self):
        """Initialize SQLite database for caching"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraped_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                content_hash TEXT,
                quality_score REAL,
                timestamp DATETIME,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                results TEXT,
                timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def acquire_data_for_topic(
        self,
        topic: str,
        domain: str = 'general',
        max_sources: int = 20,
        quality_threshold: float = 0.5
    ) -> List[ScrapedContent]:
        """
        Acquire high-quality data for a given topic
        
        Args:
            topic: Topic to research
            domain: Domain type for query generation
            max_sources: Maximum number of sources to scrape
            quality_threshold: Minimum quality score for content
        """
        logging.info(f"Acquiring data for topic: {topic}")
        
        # Generate search queries
        queries = self.search_engine.generate_search_queries(
            topic, domain, num_queries=3
        )
        
        # Search for relevant URLs
        all_urls = set()
        for query in queries:
            search_results = await self.search_engine.search(
                query, max_results=max_sources // len(queries)
            )
            
            for result in search_results:
                if result.url and result.relevance_score > 0.3:
                    all_urls.add(result.url)
        
        # Scrape content from URLs
        scraped_contents = []
        
        for i, url in enumerate(list(all_urls)[:max_sources]):
            # Rate limiting
            domain = urlparse(url).netloc
            if domain in self.last_request_time:
                elapsed = time.time() - self.last_request_time[domain]
                if elapsed < self.config.scraping_delay:
                    await asyncio.sleep(self.config.scraping_delay - elapsed)
            
            self.last_request_time[domain] = time.time()
            
            # Check cache first
            cached_content = self._get_cached_content(url)
            if cached_content:
                scraped_contents.append(cached_content)
                continue
            
            # Scrape new content
            content = await self.web_scraper.scrape_url(url)
            
            if content and content.quality_score >= quality_threshold:
                # Cache the content
                self._cache_content(content)
                scraped_contents.append(content)
                
                logging.info(f"Scraped content from {url} (quality: {content.quality_score:.2f})")
            
            # Progress logging
            if (i + 1) % 5 == 0:
                logging.info(f"Processed {i + 1}/{len(all_urls)} URLs")
        
        logging.info(f"Acquired {len(scraped_contents)} high-quality sources for '{topic}'")
        return scraped_contents
    
    def _get_cached_content(self, url: str) -> Optional[ScrapedContent]:
        """Get cached content for URL"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM scraped_content WHERE url = ?',
            (url,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return ScrapedContent(
                url=row[1],
                title=row[2],
                content=row[3],
                metadata=json.loads(row[7]),
                timestamp=datetime.fromisoformat(row[6]),
                content_hash=row[4],
                quality_score=row[5]
            )
        
        return None
    
    def _cache_content(self, content: ScrapedContent):
        """Cache scraped content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO scraped_content 
                (url, title, content, content_hash, quality_score, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                content.url,
                content.title,
                content.content,
                content.content_hash,
                content.quality_score,
                content.timestamp.isoformat(),
                json.dumps(content.metadata)
            ))
            
            conn.commit()
        except Exception as e:
            logging.error(f"Failed to cache content: {str(e)}")
        finally:
            conn.close()
    
    async def continuous_data_collection(
        self,
        topics: List[str],
        collection_interval: timedelta = timedelta(hours=6),
        max_iterations: Optional[int] = None
    ):
        """
        Continuously collect data for a list of topics
        
        Args:
            topics: List of topics to monitor
            collection_interval: Time between collection cycles
            max_iterations: Maximum number of collection cycles (None for infinite)
        """
        iteration = 0
        
        while max_iterations is None or iteration < max_iterations:
            logging.info(f"Starting data collection cycle {iteration + 1}")
            
            for topic in topics:
                try:
                    contents = await self.acquire_data_for_topic(topic)
                    logging.info(f"Collected {len(contents)} new sources for '{topic}'")
                    
                    # Optional: Process new content immediately
                    # This could trigger SEAL adaptation
                    
                except Exception as e:
                    logging.error(f"Failed to collect data for '{topic}': {str(e)}")
            
            iteration += 1
            
            if max_iterations is None or iteration < max_iterations:
                logging.info(f"Waiting {collection_interval} before next cycle")
                await asyncio.sleep(collection_interval.total_seconds())
        
        logging.info("Continuous data collection completed")
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count total content
        cursor.execute('SELECT COUNT(*) FROM scraped_content')
        total_content = cursor.fetchone()[0]
        
        # Count by quality score ranges
        cursor.execute('SELECT COUNT(*) FROM scraped_content WHERE quality_score >= 0.8')
        high_quality = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM scraped_content WHERE quality_score >= 0.5 AND quality_score < 0.8')
        medium_quality = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM scraped_content WHERE quality_score < 0.5')
        low_quality = cursor.fetchone()[0]
        
        # Average quality score
        cursor.execute('SELECT AVG(quality_score) FROM scraped_content')
        avg_quality = cursor.fetchone()[0] or 0.0
        
        # Most recent data
        cursor.execute('SELECT MAX(timestamp) FROM scraped_content')
        latest_timestamp = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_content_pieces': total_content,
            'high_quality_content': high_quality,
            'medium_quality_content': medium_quality,
            'low_quality_content': low_quality,
            'average_quality_score': avg_quality,
            'latest_data_timestamp': latest_timestamp,
            'cache_database_path': str(self.db_path)
        }
    
    def export_data(
        self,
        output_path: str,
        format: str = 'jsonl',
        quality_threshold: float = 0.5
    ):
        """
        Export collected data to file
        
        Args:
            output_path: Path to save exported data
            format: Export format ('jsonl', 'json', 'txt')
            quality_threshold: Minimum quality score for export
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM scraped_content WHERE quality_score >= ?',
            (quality_threshold,)
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for row in rows:
                    content_data = {
                        'url': row[1],
                        'title': row[2],
                        'content': row[3],
                        'quality_score': row[5],
                        'timestamp': row[6],
                        'metadata': json.loads(row[7])
                    }
                    f.write(json.dumps(content_data, ensure_ascii=False) + '\n')
        
        elif format == 'json':
            all_content = []
            for row in rows:
                content_data = {
                    'url': row[1],
                    'title': row[2],
                    'content': row[3],
                    'quality_score': row[5],
                    'timestamp': row[6],
                    'metadata': json.loads(row[7])
                }
                all_content.append(content_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_content, f, ensure_ascii=False, indent=2)
        
        elif format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for row in rows:
                    f.write(f"Title: {row[2]}\n")
                    f.write(f"URL: {row[1]}\n")
                    f.write(f"Quality: {row[5]:.2f}\n")
                    f.write(f"Content:\n{row[3]}\n")
                    f.write("-" * 80 + "\n\n")
        
        logging.info(f"Exported {len(rows)} content pieces to {output_path}")


# Example usage and testing
async def main():
    """Example usage of the data engine"""
    config = DataConfig()
    engine = DataEngine(config)
    
    # Test data acquisition
    topics = ["machine learning", "artificial intelligence", "deep learning"]
    
    for topic in topics:
        contents = await engine.acquire_data_for_topic(topic, max_sources=5)
        print(f"\nTopic: {topic}")
        print(f"Collected {len(contents)} sources")
        
        for content in contents[:2]:  # Show first 2
            print(f"  - {content.title} (Quality: {content.quality_score:.2f})")
            print(f"    {content.content[:200]}...")
    
    # Print statistics
    stats = engine.get_data_statistics()
    print(f"\nData Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())