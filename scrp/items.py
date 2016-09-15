# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy

#class ImmoData(scrapy.Item) :
#    title     = scrapy.Field()
#    info      = scrapy.Field()
#    promoteur = scrapy.Field()
#    prom_addr = scrapy.Field()
#    desc      = scrapy.Field()


class ProjectimmoItem(scrapy.Item):
    a_url = scrapy.Field()
    data  = scrapy.Field()
    #data  = ImmoData()

#    title   = scrapy.Field()
#    info    = scrapy.Field()
#    promoteur = scrapy.Field()
#    prom_addr = scrapy.Field()
#    desc      = scrapy.Field()
#
#    details   = scrapy.Field()
#
#    #ville   = scrapy.Field()
#    #adresse = scrapy.Field()
#    #nPieces = scrapy.Field()
#    #surface = scrapy.Field()
#    #prix    = scrapy.Field()

    pass
