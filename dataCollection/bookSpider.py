import scrapy
import os
import json
import requests
import re

class BookSpider(scrapy.Spider):
    name = "youGloriousLittleBookSpider"
    def start_requests(self):
        for i in range(2004,2019):
            url = 'https://www.goodreads.com/book/popular_by_date/'+str(i)+'/'
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self,response):
        base_url = 'https://www.goodreads.com'
        y = response.url
        year = y.split('/')[5]
        for book in response.css('a.bookTitle::attr(href)').extract():
            next_page = base_url+book
            yield scrapy.Request(next_page, callback = self.parseBook, meta={'year':year})

    def parseBook(self,response):
        currDir = os.getcwd()
        self.review = ''
        self.bookUrl = 'https://images.gr-assets.com/books/'

        newBook = dict()
        newBook['url']=response.url
        god = response.meta['year']

        def extractTitle(query):
            newBook['title'] = response.css(query).extract_first().strip()
            return newBook['title']
        def extractAuthor(query):
            newBook['author'] = response.css(query).extract_first().strip()
            return newBook['author']
        def extractRating(query):
            newBook['rating'] = response.css(query).extract_first().strip()
            return newBook['rating']
        def extractVotes(query):
            newBook['votes'] = response.css(query).extract_first().strip()
            return newBook['votes']
        def extractDetails(query):
            try:
                detailsText = response.css(query).extract()
                firstLine = detailsText.pop(0)
            except:
                detailsText = []
            add = False
            for dt in detailsText:
                if dt.startswith(firstLine):
                    add = True
                if add:
                    self.review = self.review + dt
            newBook['review']=self.review
            return ''
        def extractImage(query):
            firstNoFull = response.css(query).extract()[8]
            firstNo = firstNoFull.split('/')[7]
            secondNoFull = firstNoFull.split('/')[8]
            secondNo = secondNoFull.split('.')[0]
            newBook['imgUrl'] = self.bookUrl + firstNo[:-1]+'l/'+secondNo+'.jpg'
            return ''
        def extractPub(query):
            fullDetails = response.xpath(query).extract()
            for det in fullDetails:
                det = det.strip()
                if det.startswith('Published'):
                    try:
                        publisher = det.split('by')[1].strip()
                    except:
                        publisher = ''
                    try:
                        datePublished = det.split('by')[0].strip()
                    except:
                        datePublished = ''
                    newBook['datePublished'] = datePublished[10:].strip()
                    newBook['publisher'] = publisher
            return ''
        def extractIsbn(query):
            isbn = ''
            
            try:
                isbn = response.xpath(query).extract_first()
            except:
                isbn = '1'

            if isbn is None:
                isbn = response.css('div.infoBoxRowItem::text').extract()[1].strip()

            newBook['ISBN'] = isbn
            return newBook['ISBN']
        def extractAwards(query):
            awards = []
            try:
                awards = response.xpath(query).extract()
            except:
                awards = []
            newBook['awards'] = awards
            newBook['awardsNo'] = len(awards)
            return awards
        def extractRatings(query):
            ratStr = response.xpath(query).extract_first()
            extracted = (ratStr.split('['))[1].split(']')[0]
            extrSplit = extracted.split(',')
            ratDict = dict()
            ratDict['5'] = extrSplit[0]
            ratDict['4'] = extrSplit[1]
            ratDict['3'] = extrSplit[2]
            ratDict['2'] = extrSplit[3]
            ratDict['1'] = extrSplit[4]
            newBook['ratings'] = ratDict
            return ''
        def extractGenres(query):
            newBook['genres'] = response.css(query).extract()
            return ''
        def extractPages(query):
            try:
                pages = response.xpath(query).extract_first()
                newBook['pages'] = pages.split(' ')[0]
            except:
                newBook['pages'] = '0'
            return ''
        def extractAuthorUrl(query):
            newBook['authorUrl'] = response.xpath(query).extract_first()
            return ''

        yield {
            'title' : extractTitle('span.fn::text'),
            'author' : extractAuthor('div.authorName__container a.authorName span::text'),
            'rating' : extractRating('span.value span.average::text'),
            'votes' : extractVotes('a.gr-hyperlink span.votes::text'),
            'details' : extractDetails('div.readable span::text'),
            'image': extractImage('meta::attr(content)'),
            'publication' : extractPub('//div[@id="details"]/div[@class="row"]/text()'),
            'isbn' : extractIsbn('//div[@itemprop="isbn"]/text()'),
            'awards' : extractAwards('//div[@itemprop="awards"]/a/text()'),
            'ratings' : extractRatings('//script[@type="text/javascript+protovis"]'),
            'genres' : extractGenres('div.bigBoxContent div.elementList div.left a::text'),
            'pages' : extractPages('//span[@itemprop="numberOfPages"]/text()'),
            'authorUrl' : extractAuthorUrl('//div[@class="authorName__container"]/a[@itemprop="url"]/@href')
        }

        path = currDir + '/books/'+str(god)+'/'+newBook['title']

        try:  
            os.mkdir(path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)

        detailsFilename = path + '/details.json'

        with open(detailsFilename, 'w') as fp:
            json.dump(newBook, fp, sort_keys=True, indent=4)
