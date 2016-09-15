import urlparse

from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from scrapy.selector import HtmlXPathSelector
from scrapy.http import Request
from scrapy.spider import BaseSpider

from ProjectImmo.items import ProjectimmoItem

##class supimo(CrawlSpider):
class supimo(BaseSpider):
    name = 'supimo'

    #def __init__(self, region='') :
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

        urls = sel.xpath('//div[contains(@class,"title")][1]//a[contains(@target,"_blank")]/@href').extract()
        for url in urls :
            print 'SSUUUUUUUUBBBBBBBBBBBURL: ', url
            fullurl = 'http://superimmoneuf.com%s' % (url)
            yield Request(fullurl, self.parse_item)

        for url in sel.xpath('//a[@class="next paginate"]/@href').extract():
            newurl = urlparse.urljoin(response.url,url)
            print 'NEWWWWWWWWWWWWURL: ', newurl
            yield Request(urlparse.urljoin(response.url,url))
    def parse_item(self, response) :
        print 'PARSE_ITEM CALLLLLLEEEEEDDDDDD'
        sel = Selector(response)
        item = ProjectimmoItem()

        item['a_url'] = str(response.url)

        item['data'] = {}
        item['data']['title'] = str(sel.xpath("//h1/text()").extract())
        item['data']['info']  = str(sel.xpath('//p/strong/text()').extract())
        item['data']['promoteur'] = sel.xpath('//div[contains(@class, "promoter_lead")][1]//div[contains(@class, "infos")][1]/text()').extract()
        item['data']['prom_addr'] = sel.xpath('//div[contains(@class, "infos")][1]//h6/text()').extract()
        item['data']['desc']    = sel.xpath('//p[contains(@class,"description")]/text()').extract()
        item['data']['details'] = sel.xpath('//div[@class="first-col show-program"]/table/tbody/tr//text()').extract()


        yield item


