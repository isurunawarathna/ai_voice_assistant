
import asyncio
import hashlib
import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_URL = "https://www.c-clarke.com/"  # Replace with campus URL

MAX_PAGES = 100
CRAWL_DELAY = 2
PAGE_TIMEOUT = 30000
MAX_RETRIES = 3

USER_AGENT = "CClarkKnowledgeBot/1.0"

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEBSITE_JSON = OUTPUT_DIR / "website_content.json"
PDF_LINKS_JSON = OUTPUT_DIR / "pdf_links.json"

BASE_HOST = urlparse(BASE_URL).hostname

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("campus_scraper")

robots = RobotFileParser()


# ==========================================================
# URL UTILITIES
# ==========================================================

def normalize_url(url: str) -> str:
    normalized, _ = urldefrag(url)
    return normalized


def allowed_url(url: str) -> bool:
    parsed = urlparse(url)

    return (
        parsed.scheme in ("http", "https")
        and parsed.hostname == BASE_HOST
        and parsed.username is None
        and parsed.password is None
    )


def is_pdf(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def generate_hash(content: str) -> str:
    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


# ==========================================================
# ROBOTS.TXT VALIDATION
# ==========================================================

async def load_robots():
    robots_url = urljoin(BASE_URL, "/robots.txt")

    logger.info("Checking robots.txt: %s", robots_url)

    async with httpx.AsyncClient(
        timeout=15,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT}
    ) as client:

        try:
            response = await client.get(robots_url)

            if response.status_code == 200:
                robots.parse(response.text.splitlines())
                logger.info("robots.txt loaded successfully")

            elif response.status_code in (404, 410):
                robots.parse([])
                logger.warning(
                    "robots.txt not found; verify website terms"
                )

            else:
                raise RuntimeError(
                    "Cannot safely determine crawling permissions. "
                    f"robots.txt HTTP {response.status_code}"
                )

        except httpx.RequestError as error:
            raise RuntimeError(
                f"Failed to retrieve robots.txt: {error}"
            ) from error


# ==========================================================
# CONTENT EXTRACTION
# ==========================================================

def extract_page_content(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title else ""
    )

    description_tag = soup.find(
        "meta",
        attrs={"name": "description"}
    )

    description = (
        description_tag.get("content", "")
        if description_tag else ""
    )

    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript"
    ]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.body
    )

    content = (
        main.get_text("\n", strip=True)
        if main else ""
    )

    return {
        "document_id": generate_hash(url),
        "source_type": "website",
        "source_url": url,
        "title": title,
        "description": description,
        "content": content,
        "content_hash": generate_hash(content),
        "language": "unknown",
        "scraped_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "approval_status": "pending_review"
    }


def extract_links(html: str, current_url: str):
    soup = BeautifulSoup(html, "html.parser")

    page_links = set()
    pdf_links = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()

        if not href:
            continue

        absolute_url = normalize_url(
            urljoin(current_url, href)
        )

        if not allowed_url(absolute_url):
            continue

        if is_pdf(absolute_url):
            pdf_links.add(absolute_url)
        else:
            page_links.add(absolute_url)

    return page_links, pdf_links


# ==========================================================
# PAGE SCRAPER
# ==========================================================

async def scrape_page(page, url: str):
    for attempt in range(1, MAX_RETRIES + 1):

        try:
            logger.info(
                "Scraping %s (attempt %s)",
                url,
                attempt
            )

            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=PAGE_TIMEOUT
            )

            if response is None:
                logger.warning("No response: %s", url)
                return None

            if response.status in (403, 401):
                logger.warning(
                    "Access denied (%s): %s",
                    response.status,
                    url
                )
                return None

            if response.status == 429:
                logger.warning(
                    "Rate limited: %s",
                    url
                )
                return None

            if response.status >= 400:
                logger.warning(
                    "HTTP %s: %s",
                    response.status,
                    url
                )

                if response.status < 500:
                    return None

                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                    continue

                return None

            await page.wait_for_timeout(1000)

            html = await page.content()

            document = extract_page_content(html, url)

            page_links, pdf_links = extract_links(
                html,
                url
            )

            return document, page_links, pdf_links

        except PlaywrightTimeoutError:
            logger.warning("Timeout: %s", url)

        except Exception as error:
            logger.exception(
                "Failed to scrape %s: %s",
                url,
                error
            )

        if attempt < MAX_RETRIES:
            await asyncio.sleep(2 ** attempt)

    return None


# ==========================================================
# MAIN CRAWLER
# ==========================================================

async def scrape_website():

    await load_robots()

    visited = set()
    queued = {normalize_url(BASE_URL)}
    queue = deque([normalize_url(BASE_URL)])

    documents = []
    discovered_pdfs = set()

    async with async_playwright() as playwright:

        browser = await playwright.chromium.launch(
            headless=True
        )

        context = await browser.new_context(
            user_agent=USER_AGENT,
            accept_downloads=False
        )

        page = await context.new_page()

        try:
            while queue and len(visited) < MAX_PAGES:

                url = queue.popleft()
                queued.discard(url)

                if url in visited:
                    continue

                if not allowed_url(url):
                    continue

                if not robots.can_fetch(USER_AGENT, url):
                    logger.warning(
                        "Blocked by robots.txt: %s",
                        url
                    )
                    continue

                visited.add(url)

                result = await scrape_page(page, url)

                if result is None:
                    await asyncio.sleep(CRAWL_DELAY)
                    continue

                document, page_links, pdf_links = result

                if document["content"]:
                    documents.append(document)

                for pdf_url in pdf_links:
                    if robots.can_fetch(USER_AGENT, pdf_url):
                        discovered_pdfs.add(pdf_url)

                for link in page_links:

                    if link in visited or link in queued:
                        continue

                    if not robots.can_fetch(
                        USER_AGENT,
                        link
                    ):
                        continue

                    queue.append(link)
                    queued.add(link)

                logger.info(
                    "Progress: %s pages | %s PDFs",
                    len(visited),
                    len(discovered_pdfs)
                )

                await asyncio.sleep(CRAWL_DELAY)

        finally:
            await browser.close()

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    WEBSITE_JSON.write_text(
        json.dumps(
            documents,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    PDF_LINKS_JSON.write_text(
        json.dumps(
            sorted(discovered_pdfs),
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    logger.info("=" * 50)
    logger.info("SCRAPING COMPLETED")
    logger.info("Pages visited: %s", len(visited))
    logger.info("Documents saved: %s", len(documents))
    logger.info("PDF links found: %s", len(discovered_pdfs))
    logger.info("Website JSON: %s", WEBSITE_JSON)
    logger.info("PDF links JSON: %s", PDF_LINKS_JSON)
    logger.info("=" * 50)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    asyncio.run(scrape_website())
