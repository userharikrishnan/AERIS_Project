"""
AERIS Web Scraper Tool
Fetches and extracts meaningful content from web pages.
Used for: "scrape this site", "get info from this URL", "extract data from X"

Uses:
- requests: HTTP fetching
- beautifulsoup4: HTML parsing and content extraction

Output structure: title, headings, paragraphs, tables, links, metadata
"""

import time
import logging
from typing import Optional
from services.tool_base import Tool, ToolResult

logger = logging.getLogger(__name__)


class WebScraperTool(Tool):
    name = "web_scraper"
    description = "Scrape and extract structured content from web pages"
    requires_confirmation = False
    capabilities = ["web_scrape", "scrape", "extract_web_content"]

    # Max content size to prevent memory overload
    MAX_CONTENT_LENGTH = 500_000  # 500KB

    def execute(self, params: dict) -> ToolResult:
        start_time = time.time()

        url = params.get("url") or params.get("query")
        if not url:
            return ToolResult(
                success=False,
                error="Missing 'url' parameter for web scraping.",
                meta={"tool": self.name}
            )

        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }

            response = requests.get(
                url,
                headers=headers,
                timeout=15,
                allow_redirects=True
            )
            response.raise_for_status()

            # Check content length
            content = response.text
            if len(content) > self.MAX_CONTENT_LENGTH:
                content = content[:self.MAX_CONTENT_LENGTH]
                logger.warning(f"[WebScraperTool] Content truncated at {self.MAX_CONTENT_LENGTH} chars")

            soup = BeautifulSoup(content, "html.parser")

            # Remove noise elements
            for tag in soup(["script", "style", "noscript", "nav", "footer",
                              "aside", "ad", "advertisement", "cookie-banner"]):
                tag.decompose()

            # --- Extract title ---
            title = ""
            if soup.title:
                title = soup.title.get_text(strip=True)

            # --- Extract meta description ---
            meta_desc = ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            if meta_tag:
                meta_desc = meta_tag.get("content", "")

            # --- Extract headings ---
            headings = []
            for tag in ["h1", "h2", "h3", "h4"]:
                for el in soup.find_all(tag):
                    text = el.get_text(strip=True)
                    if text and len(text) > 2:
                        headings.append({"level": tag, "text": text})

            # --- Extract paragraphs ---
            paragraphs = []
            for el in soup.find_all("p"):
                text = el.get_text(strip=True)
                if text and len(text) > 30:  # Skip tiny/empty paragraphs
                    paragraphs.append(text)

            # --- Extract tables ---
            tables = []
            for table in soup.find_all("table"):
                rows = []
                for tr in table.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    if cells:
                        rows.append(cells)
                if rows:
                    tables.append(rows)

            # --- Extract links ---
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                link_text = a.get_text(strip=True)
                if href.startswith("http") and link_text:
                    links.append({"text": link_text, "url": href})

            # --- Plain text fallback ---
            plain_text = soup.get_text(separator="\n", strip=True)
            # Clean up excessive whitespace
            import re
            plain_text = re.sub(r"\n{3,}", "\n\n", plain_text)
            plain_text = plain_text[:5000]  # Limit plain text

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                data={
                    "url": url,
                    "title": title,
                    "meta_description": meta_desc,
                    "headings": headings[:20],
                    "paragraphs": paragraphs[:30],
                    "tables": tables[:5],
                    "links": links[:20],
                    "plain_text": plain_text,
                    "word_count": len(plain_text.split()),
                    "status_code": response.status_code,
                },
                meta={
                    "tool": self.name,
                    "execution_time": execution_time,
                    "content_length": len(content)
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Missing dependency: {e}. Run: pip install requests beautifulsoup4",
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )
        except Exception as e:
            logger.error(f"[WebScraperTool] Error scraping {url}: {e}")
            return ToolResult(
                success=False,
                error=f"Failed to scrape '{url}': {str(e)}",
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )
