import scrapy
from my_med_scraper.items import ProductItem

class ProductSpider(scrapy.Spider):
    name = "product_spider"
    start_urls = ["https://books.toscrape.com/"]

    def parse(self, response):
        # 1. SCRAPE DATA FIRST
        for book in response.css("article.product_pod"):
            item = ProductItem()
            item['name'] = book.css("h3 a::attr(title)").get()
            item['price'] = book.css("p.price_color::text").get()
            item['image_url'] = response.urljoin(book.css("img.thumbnail::attr(src)").get())
            yield item  # <--- ITEM GOES TO MONGODB HERE

        # 2. PAGINATE SECOND
        next_page = response.css('li.next a::attr(href)').get()
        if next_page:
            # We "yield" a Request, not an Item
            yield response.follow(next_page, callback=self.parse)