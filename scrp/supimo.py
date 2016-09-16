import urlparse

#from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.spiders import CrawlSpider, Rule
from scrapy.selector import Selector
from scrapy.selector import HtmlXPathSelector
from scrapy.http import Request

from ProjectImmo.items import ProjectimmoItem

class supimo(CrawlSpider):
    name = 'supimo'
    def __init__(self) :
        self.allowed_domains = ['superimmoneuf.com']
        self.start_urls = ['http://superimmoneuf.com/']

    def parse(self, response) :
        sel = Selector(response)

        urls = sel.xpath('//ul[@id="program-by-region"][1]/li//@href').extract()
        for url in urls :
            print '-'*20,'URL REGION: ', url
            yield Request(urlparse.urljoin(response.url,url),self.parse_page)

    def parse_page(self, response) :
        sel = Selector(response)
        urls = sel.xpath('//div[1]/h3/a/@href').extract()
        for url in urls :
            fullurl = 'http://superimmoneuf.com%s' % (url)
            yield Request(fullurl, self.parse_item)

        for url in sel.xpath('//a[@class="next paginate"]/@href').extract():
            yield Request(urlparse.urljoin(response.url,url), self.parse_page)
    def parse_item(self, response) :
        sel = Selector(response)
        item = ProjectimmoItem()

        item['a_url'] = str(response.url)

        item['data'] = {}
        item['data']['title'] = str(sel.xpath("//h1/text()").extract())
        item['data']['info']  = str(sel.xpath('//p/strong/text()').extract())
        item['data']['promoteur'] = sel.xpath('.//*[@id="documentation-holder"]/div[1]/div[2]/text()').extract()
        item['data']['prom_addr'] = sel.xpath('//div[1]/div[2]/h6/text()').extract()
        item['data']['desc']    = sel.xpath('//div[4]/p[3]/text()').extract()
        item['data']['details'] = sel.xpath('//div/table//tr//text()').extract()

        yield item


