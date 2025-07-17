#!/usr/bin/env python3
"""
Beyond RAG - Real-time Information Retrieval and Augmentation
Goes beyond traditional RAG by incorporating live data, web search, and continuous updates
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import threading
import queue
from pathlib import Path
import sqlite3
import hashlib
import re
from bs4 import BeautifulSoup
import feedparser
import asyncio
import aiohttp

class BeyondRAG:
    """Advanced RAG system with real-time updates and multiple data sources"""
    
    def __init__(self, ollama_wrapper):
        self.ollama = ollama_wrapper
        self.cache_db = Path(".jigyasa/beyond_rag_cache.db")
        self.cache_db.parent.mkdir(exist_ok=True)
        self.update_queue = queue.Queue()
        self.running = False
        
        # Data sources
        self.sources = {
            'web_search': WebSearchSource(),
            'github': GitHubSource(),
            'arxiv': ArxivSource(),
            'news': NewsSource(),
            'stackoverflow': StackOverflowSource(),
            'documentation': DocumentationSource()
        }
        
        self._init_cache_db()
        self._start_update_loop()
        
    def _init_cache_db(self):
        """Initialize cache database"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            query_hash TEXT PRIMARY KEY,
            query TEXT,
            results TEXT,
            source TEXT,
            timestamp REAL,
            relevance_score REAL,
            update_frequency INTEGER DEFAULT 3600
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_graph (
            concept TEXT PRIMARY KEY,
            related_concepts TEXT,
            sources TEXT,
            last_updated REAL,
            confidence REAL
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def _start_update_loop(self):
        """Start background update loop"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def _update_loop(self):
        """Continuously update cached information"""
        while self.running:
            # Check for stale cache entries
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            # Find entries that need updating
            stale_time = time.time() - 3600  # 1 hour
            cursor.execute('''
            SELECT query_hash, query, source FROM cache 
            WHERE timestamp < ? 
            LIMIT 10
            ''', (stale_time,))
            
            stale_entries = cursor.fetchall()
            conn.close()
            
            # Update stale entries
            for query_hash, query, source in stale_entries:
                self.update_queue.put((query, source))
                
            # Process update queue
            while not self.update_queue.empty():
                try:
                    query, source = self.update_queue.get(timeout=1)
                    self._update_single_entry(query, source)
                except queue.Empty:
                    break
                    
            time.sleep(60)  # Check every minute
            
    def _update_single_entry(self, query: str, source: str):
        """Update a single cache entry"""
        if source in self.sources:
            results = self.sources[source].search(query)
            self._cache_results(query, results, source)
            
    def search(self, query: str, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enhanced search across multiple sources"""
        if sources is None:
            sources = list(self.sources.keys())
            
        # Check cache first
        cached = self._get_cached_results(query)
        if cached and self._is_cache_fresh(cached['timestamp']):
            return cached
            
        # Parallel search across sources
        results = {}
        threads = []
        
        for source_name in sources:
            if source_name in self.sources:
                thread = threading.Thread(
                    target=lambda s, q, r: r.update({s: self.sources[s].search(q)}),
                    args=(source_name, query, results)
                )
                thread.start()
                threads.append(thread)
                
        # Wait for all searches
        for thread in threads:
            thread.join(timeout=5)
            
        # Combine and rank results
        combined_results = self._combine_results(results, query)
        
        # Cache results
        self._cache_results(query, combined_results, 'combined')
        
        # Update knowledge graph
        self._update_knowledge_graph(query, combined_results)
        
        return combined_results
        
    def _combine_results(self, results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Combine and rank results from multiple sources"""
        combined = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'summary': '',
            'key_points': [],
            'relevance_score': 0.0
        }
        
        # Aggregate results
        all_content = []
        for source, data in results.items():
            if data:
                combined['sources'][source] = data
                if isinstance(data, dict) and 'content' in data:
                    all_content.append(data['content'])
                elif isinstance(data, list):
                    all_content.extend([str(item) for item in data])
                    
        # Generate summary using Ollama
        if all_content:
            summary_prompt = f"""
            Synthesize the following information about: {query}
            
            Information from multiple sources:
            {' '.join(all_content[:5000])}  # Limit context
            
            Provide:
            1. A concise summary
            2. Key points (3-5 bullet points)
            3. Relevance score (0-1)
            """
            
            response = self.ollama.generate(summary_prompt)
            
            # Parse response
            combined['summary'] = self._extract_summary(response.text)
            combined['key_points'] = self._extract_key_points(response.text)
            combined['relevance_score'] = self._calculate_relevance(response.text, query)
            
        return combined
        
    def _cache_results(self, query: str, results: Dict[str, Any], source: str):
        """Cache search results"""
        query_hash = hashlib.sha256(f"{query}:{source}".encode()).hexdigest()
        
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO cache 
        (query_hash, query, results, source, timestamp, relevance_score)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            query_hash,
            query,
            json.dumps(results),
            source,
            time.time(),
            results.get('relevance_score', 0.5)
        ))
        
        conn.commit()
        conn.close()
        
    def _get_cached_results(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached results if available"""
        query_hash = hashlib.sha256(f"{query}:combined".encode()).hexdigest()
        
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT results, timestamp FROM cache 
        WHERE query_hash = ?
        ''', (query_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'data': json.loads(result[0]),
                'timestamp': result[1]
            }
        return None
        
    def _is_cache_fresh(self, timestamp: float, max_age: int = 3600) -> bool:
        """Check if cache is fresh enough"""
        return (time.time() - timestamp) < max_age
        
    def _update_knowledge_graph(self, query: str, results: Dict[str, Any]):
        """Update knowledge graph with new information"""
        # Extract concepts from results
        concepts = self._extract_concepts(results)
        
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        for concept in concepts:
            cursor.execute('''
            INSERT OR REPLACE INTO knowledge_graph
            (concept, related_concepts, sources, last_updated, confidence)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                concept['name'],
                json.dumps(concept['related']),
                json.dumps(concept['sources']),
                time.time(),
                concept['confidence']
            ))
            
        conn.commit()
        conn.close()
        
    def _extract_summary(self, text: str) -> str:
        """Extract summary from response"""
        # Simple extraction - can be enhanced
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'summary' in line.lower():
                return ' '.join(lines[i+1:i+4])
        return text[:200]
        
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from response"""
        points = []
        lines = text.split('\n')
        for line in lines:
            if re.match(r'^\s*[-•*]\s+', line):
                points.append(line.strip(' -•*'))
        return points[:5]
        
    def _calculate_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
            
        overlap = len(query_words & text_words)
        return min(overlap / len(query_words), 1.0)
        
    def _extract_concepts(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract concepts from results"""
        concepts = []
        
        # Simple concept extraction - can be enhanced with NLP
        text = results.get('summary', '') + ' '.join(results.get('key_points', []))
        
        # Extract noun phrases (simplified)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for word in set(words):
            concepts.append({
                'name': word,
                'related': [],  # Would need NLP to find related concepts
                'sources': list(results.get('sources', {}).keys()),
                'confidence': 0.7
            })
            
        return concepts
        
    def update_knowledge(self, query: str, response: str):
        """Update knowledge base with new information"""
        # This is called after interactions to continuously learn
        self.update_queue.put((query, 'interaction'))
        
    def stop(self):
        """Stop the update loop"""
        self.running = False
        if hasattr(self, 'update_thread'):
            self.update_thread.join()

# Data source implementations
class WebSearchSource:
    """Web search data source"""
    
    def search(self, query: str) -> Dict[str, Any]:
        # Placeholder - would integrate with search API
        return {
            'content': f"Web search results for: {query}",
            'urls': [],
            'snippets': []
        }

class GitHubSource:
    """GitHub data source"""
    
    def search(self, query: str) -> Dict[str, Any]:
        try:
            # Search GitHub repositories
            response = requests.get(
                f"https://api.github.com/search/repositories",
                params={'q': query, 'sort': 'stars', 'per_page': 5},
                headers={'Accept': 'application/vnd.github.v3+json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'content': f"Found {data['total_count']} repositories",
                    'repositories': [
                        {
                            'name': repo['full_name'],
                            'description': repo['description'],
                            'stars': repo['stargazers_count'],
                            'url': repo['html_url']
                        }
                        for repo in data['items']
                    ]
                }
        except:
            pass
        return {}

class ArxivSource:
    """ArXiv papers data source"""
    
    def search(self, query: str) -> Dict[str, Any]:
        try:
            # Search ArXiv
            url = f"http://export.arxiv.org/api/query?search_query={query}&max_results=5"
            feed = feedparser.parse(url)
            
            papers = []
            for entry in feed.entries:
                papers.append({
                    'title': entry.title,
                    'authors': [a.name for a in entry.authors],
                    'summary': entry.summary[:200],
                    'link': entry.link
                })
                
            return {
                'content': f"Found {len(papers)} papers",
                'papers': papers
            }
        except:
            pass
        return {}

class NewsSource:
    """News data source"""
    
    def search(self, query: str) -> Dict[str, Any]:
        # Placeholder - would integrate with news API
        return {
            'content': f"Latest news about: {query}",
            'articles': []
        }

class StackOverflowSource:
    """Stack Overflow data source"""
    
    def search(self, query: str) -> Dict[str, Any]:
        try:
            # Search Stack Overflow
            response = requests.get(
                "https://api.stackexchange.com/2.3/search",
                params={
                    'order': 'desc',
                    'sort': 'relevance',
                    'intitle': query,
                    'site': 'stackoverflow',
                    'pagesize': 5
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'content': f"Found {len(data.get('items', []))} questions",
                    'questions': [
                        {
                            'title': q['title'],
                            'link': q['link'],
                            'score': q['score'],
                            'answered': q['is_answered']
                        }
                        for q in data.get('items', [])
                    ]
                }
        except:
            pass
        return {}

class DocumentationSource:
    """Documentation data source"""
    
    def search(self, query: str) -> Dict[str, Any]:
        # Placeholder - would index and search documentation
        return {
            'content': f"Documentation search for: {query}",
            'results': []
        }