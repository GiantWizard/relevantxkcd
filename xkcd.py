import scrapy
from scrapy.crawler import CrawlerProcess
import sys
import os

sys.dont_write_bytecode = True

class XkcdItem(scrapy.Item):
    comic_id = scrapy.Field()
    text_content = scrapy.Field()

class XkcdPipeline:
    def __init__(self):
        self.file_path = "explanations.txt"
        self.file = None

    def openSpider(self, spider):
        self.file = open(self.file_path, "a", encoding="utf-8")

    def closeSpider(self, spider):
        if self.file:
            self.file.close()

    def processItem(self, item, spider):
        comicID = item["comic_id"]
        content = item["text_content"]

        self.file.write(f"{comicID}:\n{content}\n---\n")
        self.file.flush()
        spider.processed.add(comicID)
        return item

class XkcdSpider(scrapy.Spider):
    name = "xkcd"
    
    custom_settings = {
        'ITEM_PIPELINES': {'__main__.XkcdPipeline': 1},
        'DOWNLOAD_DELAY': 0.5, 
        'CONCURRENT_REQUESTS_PER_DOMAIN': 4,
        'LOG_LEVEL': 'INFO' 
    }

    def __init__(self, *args, **kwargs):
        super(XkcdSpider, self).__init__(*args, **kwargs)
        self.processed = self.getProcessed()

    def getProcessed(self):
        path = "explanations.txt"
        if not os.path.exists(path):
            return set()

        processed = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.endswith(":") and line[:-1].isdigit():
                    processed.add(line[:-1])
        return processed

    def startRequests(self):
        yield scrapy.Request("https://xkcd.com/info.0.json", callback=self.parseAPI)

    def parseAPI(self, response):
        total = response.json().get("num", 0)
        self.logger.info(f"Total comics found: {total}. Resuming from saved state...")

        for i in range(1, total + 1):
            comicID = str(i)
            if comicID not in self.processed:
                url = f"https://www.explainxkcd.com/wiki/index.php/{i}"
                yield scrapy.Request(url=url, callback=self.parse, cb_kwargs={'comic_id': comicID})

    def parseSection(self, response, sectionTitle):
        # extract text from various tags until the next header
        paragraphs = []
        header = response.xpath(f'//span[@id="{sectionTitle}"]/parent::*')

        if not header:
            return []

        for sib in header.xpath("following-sibling::*"):
            tag = sib.root.tag.lower()

            # stop if hitting a header
            if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                break

            # different tags
            if tag in ["p", "dl", "ul", "ol"]:
                # get all text and strip whitespace
                text = " ".join(sib.css("*::text").getall())
                # Clean up weird spacing 
                text = " ".join(text.split()).strip() 
                if text:
                    paragraphs.append(text)

        return paragraphs

    def parse(self, response, comicID):
        # grab alt text
        alt_text_node = response.xpath('//a[@title="Title text"]/parent::i/parent::span/following-sibling::text()').get()
        alt_text = alt_text_node.strip() if alt_text_node else ""

        # grab sections
        transcript = self.parseSection(response, "Transcript")
        explanation = self.parseSection(response, "Explanation")
        discussion = self.parseSection(response, "Discussion")

        def formatSection(name, contentList):
            if not contentList:
                return ""
            if isinstance(contentList, list):
                content = "\n".join(contentList)
            else:
                content = contentList
            return f"{name}:\n{content}\n\n"

        full_text = (
            formatSection("Alt Text", alt_text) +
            formatSection("Transcript", transcript) +
            formatSection("Explanation", explanation) +
            formatSection("Discussion", discussion)
        ).strip()

        if full_text:
            yield XkcdItem(comic_id=comicID, text_content=full_text)

if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(XkcdSpider)
    process.start()