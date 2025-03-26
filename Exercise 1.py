import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from urllib.parse import urlencode
import hashlib
from datetime import datetime

# Configure settings
settings = get_project_settings()
settings.update({
    "DOWNLOAD_HANDLERS": {
        "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    },
    "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
    "PLAYWRIGHT_BROWSER_TYPE": "chromium",
    "DOWNLOADER_MIDDLEWARES": {
        "scrapy_playwright.middleware.PlaywrightMiddleware": 800,
    },
    "CONCURRENT_REQUESTS": 4,
    "AUTOTHROTTLE_ENABLED": True,
})


class IndeedSpider(scrapy.Spider):
    name = "indeed_jobs"
    base_url = "https://www.indeed.com/jobs?"

    def start_requests(self):
        params = {
            "q": "software engineer",
            "l": "remote",
            "fromage": "7"  # Last 7 days
        }
        url = self.base_url + urlencode(params)
        yield scrapy.Request(
            url,
            meta={"playwright": True},
            callback=self.parse_job_list,
        )

    def parse_job_list(self, response):
        for job in response.css('div.job_seen_beacon'):
            job_url = job.css('h2 a::attr(href)').get()
            if job_url:
                yield response.follow(
                    job_url,
                    meta={"playwright": True},
                    callback=self.parse_job_page,
                )

        # Pagination
        next_page = response.css('a[data-testid="pagination-page-next"]::attr(href)').get()
        if next_page:
            yield response.follow(
                next_page,
                meta={"playwright": True},
                callback=self.parse_job_list,
            )

    def parse_job_page(self, response):
        def clean_text(text):
            return " ".join(text.strip().split()) if text else None

        yield {
            "job_id": hashlib.sha256(response.url.encode()).hexdigest(),
            "source": "indeed",
            "job_title": clean_text(response.css('h1.jobTitle::text').get()),
            "company": clean_text(response.css('div.companyInfoContainer span::text').get()),
            "location": clean_text(response.css('div.companyLocation::text').get()),
            "salary": clean_text(response.css('span.estimated-salary-content span::text').get()),
            "description": clean_text(" ".join(response.css('div#jobDescriptionText *::text').getall())),
            "posted_date": clean_text(response.css('span.date::text').get()),
            "url": response.url,
            "timestamp": datetime.utcnow().isoformat(),
        }


if __name__ == "__main__":
    process = CrawlerProcess(settings)
    process.crawl(IndeedSpider)
    process.start()
