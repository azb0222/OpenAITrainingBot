from threading import Thread

import asyncio
from asyncio import Semaphore
from tqdm.asyncio import tqdm as async_tqdm
from typing import Any, Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document




from playwright.async_api import async_playwright
INNER_TEXT = "node => node.innerText"


async def get_page_urls(browser: Any, url: str, link_selectors, root_url, semaphore, progress_bar):
    async with semaphore:
        page = await browser.new_page(ignore_https_errors=True)
        page.set_default_timeout(60000)
        await page.goto(url, wait_until="domcontentloaded")

        links = []
        for link_selector in link_selectors:
            ahrefs = await page.query_selector_all(link_selector)
            for link in ahrefs:
                href = await link.evaluate("node => node.href")
                if href.startswith(root_url) and 'firmware/' not in href:
                    links.append(href)

        await page.close()
        progress_bar.update(1)
        return links


class KnowledgeBaseWebReader(BaseReader):
    """Knowledge base reader.
    Crawls and reads articles from a knowledge base/help center with Playwright.
    Tested on Zendesk and Intercom CMS, may work on others.
    Can be run in headless mode but it may be blocked by Cloudflare. Run it headed to be safe.
    Times out occasionally, just increase the default time out if it does.
    Requires the `playwright` package.
    Args:
        root_url (str): the base url of the knowledge base, with no trailing slash
            e.g. 'https://support.intercom.com'
        link_selectors (List[str]): list of css selectors to find links to articles while crawling
            e.g. ['.article-list a', '.article-list a']
        title_selector (Optional[str]): css selector to find the title of the article
            e.g. '.article-title'
        subtitle_selector (Optional[str]): css selector to find the subtitle/description of the article
            e.g. '.article-subtitle'
        body_selector (Optional[str]): css selector to find the body of the article
            e.g. '.article-body'
    """

    def __init__(
        self,
        root_url: str,
        link_selectors: List[str],
        title_selector: Optional[str] = None,
        subtitle_selector: Optional[str] = None,
        body_selector: Optional[str] = None,
    ) -> None:
        """Initialize with parameters."""
        self.root_url = root_url
        self.link_selectors = link_selectors
        self.title_selector = title_selector
        self.subtitle_selector = subtitle_selector
        self.body_selector = body_selector

    async def load_data(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            # Crawl
            article_urls = await self.get_article_urls(browser, self.root_url)

            # Scrape
            scrape_tasks = []
            semaphore = asyncio.Semaphore(16)  # Limit concurrency to 16
            with async_tqdm(total=len(article_urls), desc="Scraping Articles") as progress_bar:  # Add the progress bar
                for url in article_urls:
                    task = asyncio.create_task(self._scrape_article_with_semaphore(semaphore, browser, url, progress_bar))
                    scrape_tasks.append(task)

                scraped_articles = await asyncio.gather(*scrape_tasks)

            documents = dict()
            categories = set()
            for article in scraped_articles:
                extra_info = {
                    "title": article["title"],
                    "category": article["category"],
                    "url": article["url"],
                }
                categories.add(article["category"])
                if article["category"] in documents:
                    documents[article["category"]].append(Document(text=article["body"], extra_info=extra_info))
                else:
                    documents[article["category"]] = [Document(text=article["body"], extra_info=extra_info)]

            await browser.close()

            return (documents, categories)

    async def _scrape_article_with_semaphore(self, semaphore, browser, url, progress_bar):
        async with semaphore:
            result = await self.scrape_article(browser, url)
            progress_bar.update(1)  # Update the progress bar
            return result

    async def scrape_article(
        self,
        browser: Any,
        url: str,
    ) -> Dict[str, str]:
        """Scrape a single article url.
        Args:
            browser (Any): a Playwright Chromium browser.
            url (str): URL of the article to scrape.
        Returns:
            Dict[str, str]: a mapping of article attributes to their values.
        """
        page = await browser.new_page(ignore_https_errors=True)  # Added 'await' here
        page.set_default_timeout(60000)
        await page.goto(url, wait_until="domcontentloaded")
        title = url.split("/")[-1].replace("-", " ").title()
        category = url.split("/")[-2].replace("-", " ").title()
        body = (
            (
                await (await page.query_selector(self.body_selector)).evaluate(
                    INNER_TEXT
                )
            )
            if self.body_selector
            else ""
        )

        await page.close()
        return {"title": title, "category": category, "body": body, "url": url}

    async def get_article_urls(self, browser: Any, root_url: str):
        article_urls = set()
        article_urls.add(root_url)
        visited_urls = []
        semaphore = Semaphore(16)

        urls = [root_url]

        with async_tqdm(total=len(urls), desc="Scraping URLs") as progress_bar:  # Add the progress bar
            while len(article_urls.difference(visited_urls)) > 0:
                # Update the progress bar
                progress_bar.total = len(article_urls)
                
                urls = article_urls.difference(visited_urls)

                tasks = []
                for url in urls:
                    task = asyncio.create_task(get_page_urls(browser, url, self.link_selectors, self.root_url, semaphore, progress_bar))
                    tasks.append(task)
                results = await asyncio.gather(*tasks)

                for result in results:
                    article_urls.update(result)
                visited_urls.extend(urls)


        return article_urls