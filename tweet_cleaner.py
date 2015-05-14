__author__ = 'Pablo'


from util import tweet_preprocesser
from util import retrieve_files
from util import load_json
from util import save_json
from unidecode import unidecode


def process_tweets(tweets, opath, oname, hashtag=None, exceptions=[]):
    """
    Drops retweets, repeated tweets and, if provided, tweets out of hashtag.
    :param tweets: file containing tweets
    :param opath:
    :param oname:
    :param hashtag:
    :param exceptions: tweets ids that are considered even if they are not in the hashtag.
    :return:
    """
    processed_tweets = [None] * len(tweets)

    k = 0
    retweets = 0
    out_of_hashtag = 0
    repeated = 0

    ids = []

    for tweet in tweets:

        if not 'retweeted_status' in tweet.keys():

            if tweet['id'] in ids:
                repeated += 1
                continue

            if hashtag is None or hashtag.lower() in [unidecode(hash['text'].lower()) for hash in
                                                      tweet['entities']['hashtags']] or tweet['id'] in exceptions:
                processed_tweets[k] = tweet
                ids.append(tweet['id'])

                k += 1
            else:
                out_of_hashtag += 1

        else:
            retweets += 1
            #print u"Retweet found ien tweet file:\n{0}".format(tweet['text'])

    ofilename = '{0}/{1}'.format(opath, oname)
    save_json(ofilename, processed_tweets[0:k])

    print "File {0} containing tweets processed succesfully.\n\tNumber of retweets found and skipped: {1}." \
          "\n\tNumber of tweets outside hashtag {2}\n\t Repeated: {3}\n".format(oname, retweets, out_of_hashtag,
                                                                                repeated)


def process_retweets(tweets, opath, oname, hashtag=None, exceptions=[]):
    """
    Drops tweets, repeated retweets and, if provided, tweets out of hashtag.

    :param tweets:
    :param opath:
    :param oname:
    :param hashtag:
    :param exceptions:
    :return: retweets ids that are considered even if they are not in the hashtag.
    """
    processed_tweets = [None] * len(tweets)

    ids = []

    k = 0
    n_tweets = 0
    out_of_hashtag = 0
    repeated = 0

    for tweet in tweets:

        if 'retweeted_status' in tweet.keys():

            if tweet['id'] in ids:
                repeated += 1
                continue

            if hashtag is None or hashtag.lower() in [unidecode(hash['text'].lower()) for hash in
                                    tweet['entities']['hashtags']] or tweet['retweeted_status']['id'] in exceptions:
                processed_tweets[k] = tweet
                ids.append(tweet['id'])
                k += 1
            else:
                out_of_hashtag += 1
        else:
            n_tweets += 1
            #print u"Tweet found in retweet file: {0}".format(tweet['text'])

    ofilename = '{0}/{1}'.format(opath, oname)
    save_json(ofilename, processed_tweets[0:k])

    print "File {0} containing tweets processed succesfully.\n\tNumber of retweets found and skipped: {1}." \
          "\n\tNumber of tweets outside hashtag {2}\n\t Repeated: {3}\n".format(oname, n_tweets, out_of_hashtag,
                                                                                repeated)


def main():


    path = 'original_tweets'
    opath = 'cleaned_tweets/by_hashtag'
    files = retrieve_files(path)
    exceptions = [[]]* len(files)

    for i, tweet_file in enumerate(files):

        tweet_file = tweet_file[0:-5]

        tweets = load_json('{0}/{1}'.format(path, tweet_file))
        #clean tweets and order them by timestamp
        tweets = tweet_preprocesser(tweets)

        oname = '{0}_cleaned'.format(tweet_file)

        print "Processing file: {0}".format(files[i])

        #If only contains retweets
        if tweet_file.endswith('_retweets'):
            process_retweets(tweets, opath, oname, exceptions=exceptions[i])
        else:
            process_tweets(tweets, opath, oname, exceptions=exceptions[i])
            #If the file contains tweets retreive real time it contains both tweets and retweets
            if tweet_file in ['IgualesPodemos', 'Podemos22M', 'SoloConTuAyuda', 'VotaPodemosAndalucia', 'YoVoy20M',
                              'CocinaCasera']:
                oname = '{0}_retweets_cleaned'.format(tweet_file)
                process_retweets(tweets, opath, oname, exceptions=exceptions[i])


if __name__ == '__main__':
    main()