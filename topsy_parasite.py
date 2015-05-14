#!/usr/bin/env python
from contextlib import closing
from selenium.webdriver import Firefox
import BeautifulSoup
import re
from util import save_json


def main():

    filename = 'topsy_results/#AsambleaCiudadana'
    #Search to do in topsy
    #QUERY= search query
    #Optional variables:
    #window=: window search: 'realtime' (latest results), 'h' (past hour), 'd' (Past 1 Day),
    #         'w' (Past 7 days), 'd27' (Past 27 days), 'm', (Past 30 days), 'a' (all time), 
    #To perform range search, in stead of window use: mintime=TIMESTAMP&maxtime=TIMESTAMP
    #sort= 'date' (newest), '-date' (oldest). Do not include this variable for sorting by revelance
    #type= 'link' (links), 'tweet' (tweet), 'image' (photos), 'video' (videos),
    #       'expert' (influencers). Do not include this variable for include Everything
    #language= 'en', 'zh', 'ja', 'ko', 'ru', 'de', 'es', 'fr', 'pt', 'tr'
    #IMPORTANT: end the url variable with string 'offset='
    url = 'http://topsy.com/s?q=%23AsambleaCiudadana&window=a&sort=-date&offset='

    #List of recovered tweets id.
    loot = []
    total_tweets = 0

    # use firefox to get page with javascript generated content
    with closing(Firefox()) as browser:
        #Loops over all possible pages by using offset variable in url
        for i in xrange(0, 1000, 10):
            browser.get(url + str(i))
            page_source = browser.page_source

            #Parser looks for tweet id
            expression = re.compile('tweet_id=([0-9]*)"')
            soup = BeautifulSoup.BeautifulSoup(page_source)
            #Looks for label <div class=result-tweet> which is the container for each tweet
            tweets = soup.findAll('div', {'class': 'result-tweet'})
            total_tweets += len(tweets)

            for tweet in tweets:
                tweet_body = tweet.find('div', {'class': 'media-body'})
                #tweet_content =  tweet_body.div.contents[0]
                #author =  tweet_body.small.contents[0]
                #Tweet id is got from the field 'retweet'
                tweet_id = expression.findall(str(tweet_body))[0]
                loot.append(tweet_id)


    save_json(filename, loot)

if __name__ == '__main__':
    main()
