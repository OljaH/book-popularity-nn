import scrapy
import pymongo

class CommentSpider(scrapy.Spider):
    name = "youBeautifulCommentSpider"
    
    def start_requests(self):
        for i in range(2005,2019):
            url = 'https://www.goodreads.com/book/popular_by_date/'+str(i)+'/'
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self,response):
        base_url = 'https://www.goodreads.com'
        for book in response.css('a.bookTitle::attr(href)').extract():
            next_page = base_url+book
            yield scrapy.Request(next_page, callback = self.parseComments)

    def parseComments(self,response):
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["SIAP"]

        mycol = mydb["comments"]

        def extractComments(query):
            myList = []
            for com in response.xpath(query):
                myList.append(com.xpath('string(.)').extract())

            adding = True
            for com in myList:
                for com2 in myList:
                    if com2[0].startswith(com[0]) and com[0] != com2[0]:
                        adding = False
                
                if adding:
                    comment = dict()
                    comment['sentiment'] = -1
                    comment['text'] = com[0]
                    comment['book'] = response.url
                    x = mycol.insert_one(comment)

                adding = True

        yield {
            'comments' : extractComments('//div[@class="reviewText stacked"]/span[@class="readable"]/span')
        }

    
