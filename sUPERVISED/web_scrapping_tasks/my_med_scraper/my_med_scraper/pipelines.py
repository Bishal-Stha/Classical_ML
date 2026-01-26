# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import pymongo
import scrapy

class MongoPipeline:
    collection_name = 'products'

    def __init__(self, mongo_uri, mongo_db):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db

    @classmethod
    def from_crawler(cls, crawler):
        # Pulls settings from settings.py
        return cls(
            mongo_uri=crawler.settings.get('MONGO_URI'),
            mongo_db=crawler.settings.get('MONGO_DATABASE')
        )

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]

    def close_spider(self, spider):
        self.client.close()

    def process_item(self, item, spider):
    # Search for a document with the same URL
        existing_item = self.db[self.collection_name].find_one({"url": item["url"]})
        
        if existing_item:
            # If it exists, tell Scrapy to drop it (ignore it)
            raise scrapy.exceptions.DropItem(f"Duplicate item found: {item['name']}") # type: ignore
        else:
            # If it's new, insert it
            self.db[self.collection_name].insert_one(dict(item))
            return item