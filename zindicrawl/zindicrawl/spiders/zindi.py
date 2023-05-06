import scrapy
import os
import html2text
from urllib.parse import urljoin


class ZindiSpider(scrapy.Spider):
    name = "zindi"
    allowed_domains = ["zindi.africa"] #controlled scrapping
    start_urls = ["https://zindi.africa/"]

    def __init__(self, *args, **kwargs):
        super(ZindiSpider, self).__init__(*args, **kwargs)

        #create the output folder if it doesn't exist
        if not os.path.exists("output"):
            os.makedirs("output")

        #Initialize the html2text converter
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = True
        self.converter.ignore_images = True
        self.converter.ignore_emphasis = True
        self.converter.ignore_tables = True
        self.converter.body_width = 0

    def parse(self, response):
        #convert the HTML content to plain text
        text = self.converter.handle(response.body.decode())
        text = text.replace("<|endoftext|>"," ")
        #clean up the text
        text = text.strip()

        #Determine the filename based on the url
        url = response.url.strip("/")
        #generate a random filename using hash of the url
        filename = f"output/{hash(url)}.txt"

        #write the text to a separate file
        with open(filename, "w") as f:
            f.write(text)

        #follow links to other pages: I
        for link in response.xpath('//a/@href'):
            href = link.get()
            # if href.startswith("/docs/diffusers"):# CONTROLLED Scraping, you can remove this
                #join the relative URL with the base URL of the current page
            url = urljoin(response.url, href)
            yield scrapy.Request(url, callback= self.parse)
