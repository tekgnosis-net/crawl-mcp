"""
SearXNG client for external search functionality.
"""

import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlencode, urlparse
from collections import defaultdict

import aiohttp


class SearXNGClient:
    """
    Client for interacting with an external SearXNG instance.

    SearXNG is a privacy-respecting metasearch engine that aggregates
    results from multiple search engines.
    """

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the SearXNG client.

        Args:
            base_url: The base URL of the SearXNG instance (e.g., "https://searx.org")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def search(
        self,
        query: str,
        num_results: int = 10,
        language: str = 'en',
        region: str = 'us',
        safe_search: bool = False,
        search_genre: Optional[str] = None,
        include_snippets: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a search using the SearXNG instance.

        Args:
            query: The search query
            num_results: Number of results to return
            language: Language code (e.g., "en", "fr")
            region: Region code (e.g., "us", "uk")
            safe_search: Enable safe search
            search_genre: Search genre/category (maps to SearXNG categories)
            include_snippets: Whether to include snippets in results

        Returns:
            Dictionary containing search results in GoogleSearchProcessor format
        """
        if not self.session:
            raise RuntimeError("Client must be used as async context manager")

        try:
            # Map search_genre to SearXNG categories
            categories = None
            if search_genre:
                genre_to_category = {
                    'general': 'general',
                    'news': 'news',
                    'videos': 'videos',
                    'images': 'images',
                    'music': 'music',
                    'social_media': 'social media',
                    'academic': 'science',
                    'shopping': 'shopping'
                }
                categories = genre_to_category.get(search_genre.lower(), search_genre)

            params = {
                'q': query,
                'format': 'json',
                'language': language,
                'safesearch': 1 if safe_search else 0,
            }

            if categories:
                params['categories'] = categories

            # SearXNG doesn't have direct region parameter, but we can try
            if region:
                params['region'] = region

            # Request more results than needed since SearXNG may return fewer
            params['num'] = min(50, num_results * 2)

            search_url = urljoin(self.base_url + '/', 'search')
            full_url = f"{search_url}?{urlencode(params)}"

            async with self.session.get(full_url) as response:
                response.raise_for_status()
                try:
                    data = await response.json()
                except aiohttp.ContentTypeError as e:
                    return {
                        'success': False,
                        'error': f'Invalid JSON response from SearXNG: {e}',
                        'query': query
                    }

            # Process SearXNG results into GoogleSearchProcessor format
            search_results = []
            searxng_results = data.get('results', [])

            for i, result in enumerate(searxng_results[:num_results]):
                try:
                    url = result.get('url', '')
                    if not url:
                        continue

                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc

                    title = result.get('title', 'No title')
                    snippet = result.get('content', 'No description available') if include_snippets else ''

                    search_result = {
                        'rank': i + 1,
                        'url': url,
                        'domain': domain,
                        'title': title,
                        'snippet': snippet,
                        'type': self._classify_url(url)
                    }

                    # Add additional metadata if available
                    if 'engine' in result:
                        search_result['engine'] = result['engine']
                    if 'engines' in result:
                        search_result['engines'] = result['engines']

                    search_results.append(search_result)

                except Exception:
                    continue

            if not search_results:
                return {
                    'success': False,
                    'error': 'No search results found',
                    'query': query,
                    'suggestion': 'Try a different search query or check SearXNG instance'
                }

            # Generate search statistics
            domains = [result['domain'] for result in search_results]
            unique_domains = list(set(domains))
            domain_counts = {domain: domains.count(domain) for domain in unique_domains}

            # Classify result types
            type_counts = {}
            for result in search_results:
                result_type = result['type']
                type_counts[result_type] = type_counts.get(result_type, 0) + 1

            return {
                'success': True,
                'query': query,
                'total_results': len(search_results),
                'results': search_results,
                'search_metadata': {
                    'search_params': {
                        'num_results_requested': num_results,
                        'language': language,
                        'region': region,
                        'safe_search': safe_search,
                        'search_genre': search_genre,
                        'include_snippets': include_snippets
                    },
                    'result_stats': {
                        'total_results': len(search_results),
                        'unique_domains': len(unique_domains),
                        'domain_distribution': domain_counts,
                        'result_types': type_counts
                    }
                },
                'processing_method': 'searxng'
            }

        except aiohttp.ClientError as e:
            return {
                'success': False,
                'error': f'Network error: {str(e)}',
                'query': query,
                'suggestion': 'Check SearXNG instance URL and network connection'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'SearXNG search error: {str(e)}',
                'query': query
            }

    def _classify_url(self, url: str) -> str:
        """Classify URL by type based on domain and path"""
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc
            path = parsed.path

            # Social media platforms
            if any(social in domain for social in ['youtube.com', 'youtu.be']):
                return 'video'
            elif any(social in domain for social in ['twitter.com', 'x.com', 'facebook.com', 'linkedin.com']):
                return 'social_media'
            elif any(social in domain for social in ['reddit.com', 'quora.com', 'stackoverflow.com']):
                return 'forum'

            # News and media
            elif any(news in domain for news in ['bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com']):
                return 'news'

            # Academic and education
            elif any(edu in domain for edu in ['.edu', '.ac.', 'scholar.google', 'arxiv.org']):
                return 'academic'

            # Government and official
            elif any(gov in domain for gov in ['.gov', '.mil', '.org']):
                return 'official'

            # E-commerce
            elif any(shop in domain for shop in ['amazon.com', 'ebay.com', 'etsy.com']):
                return 'ecommerce'

            # Documentation
            elif any(doc in domain for doc in ['github.com', 'docs.', 'wiki']):
                return 'documentation'

            # File types
            elif any(filetype in path for filetype in ['.pdf', '.doc', '.ppt']):
                return 'document'

            else:
                return 'general'

        except Exception:
            return 'unknown'

    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None


# Convenience function for one-off searches
async def search_searxng(
    base_url: str,
    query: str,
    num_results: int = 10,
    language: str = 'en',
    region: str = 'us',
    safe_search: bool = True,
    search_genre: Optional[str] = None,
    include_snippets: bool = True
) -> Dict[str, Any]:
    """
    Perform a single search using SearXNG.

    Args:
        base_url: The base URL of the SearXNG instance
        query: The search query
        num_results: Number of results to return
        language: Language code
        region: Region code
        safe_search: Enable safe search
        search_genre: Search genre/category
        include_snippets: Whether to include snippets

    Returns:
        Dictionary containing search results in GoogleSearchProcessor format
    """
    async with SearXNGClient(base_url) as client:
        return await client.search(
            query=query,
            num_results=num_results,
            language=language,
            region=region,
            safe_search=safe_search,
            search_genre=search_genre,
            include_snippets=include_snippets
        )