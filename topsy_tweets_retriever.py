import json, io
import twitter
from robust_request import make_twitter_request
import sys


def load_json(filename):
    with io.open('{0}.json'.format(filename), 
                 encoding='utf-8') as f:
        return json.load(f)


def save_json(filename, data):
    with io.open('{0}.json'.format(filename), 
                 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(data, ensure_ascii=False)))


def get_auth(user):

    if user == 'pablo':
        #tweet API connection
        CONSUMER_KEY = ''
        CONSUMER_SECRET = ''
        OAUTH_TOKEN = ''
        OAUTH_TOKEN_SECRET = ''

    else:
        raise "Unknown user"

    return CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET


def main():

    #File containing the file generated by topsy_parasite.py
    filename =  'topsy_results/#AsambleaCiudadana' #sys.argv[1]#'topsy_results/#NoOlvidamosPPSOE_topsy'
    tweets_id = load_json(filename)
    #File name for saving the tweets information.
    output_filename = 'original_tweets/#AsambleaCiudadana'#sys.argv[2]#'tweets/#NoOlvidamosPPSOE'

    user = 'alo_balles'
    print "user: {0}".format(user)
   
    CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET = get_auth(user)

    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)

    twitter_api = twitter.Twitter(auth=auth)

    total_tweets = len(tweets_id)
    retrieved_tweets = [None] * total_tweets
    #For each tweet id retrieve the tweet
    errors = 0
    for i, tweet in enumerate(tweets_id):
        try:
            retrieved_tweets[i] = make_twitter_request(twitter_api.statuses.show, id=tweet)
        except:
            print "ERROR at tweet number {0} with id {1}".format(i, tweet)
            errors += 1
        if i % 100 == 0:
            print "{0} tweets retrieved: {1} % of total tweets".format(i*100+1,
                float(i+1)/total_tweets*100)
    save_json(output_filename, retrieved_tweets)
    print "Finished with {0} errors".format(errors)

if __name__ == '__main__':
    main()