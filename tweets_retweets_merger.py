__author__ = 'Pablo'


from os import listdir
from util import tweet_preprocesser
from util import load_json
from util import save_json


def merger(tweets, retweets, ofilename, opath='merged_tweets'):
    """
    Merges tweet and retweet files into one per hashtag.
    :param tweets: json file containing tweets of one hashtag
    :param retweets: json file containing retweets of one hashtag
    :param ofilename: filename of the output file
    :param opath: path of output file
    :return:
    """

    ids = [tweet['id'] for tweet in tweets]
    repeated = 0
    for rt in retweets:
        if rt['id'] not in ids:
            ids.append(rt['id'])
            tweets.append(rt)
        else:
            repeated += 1
    print "Repeated tweets: {0}".format(repeated)
    n = len(tweets)
    tweets = tweet_preprocesser(tweets, clipper=-1)
    assert n==len(tweets)
    save_json('{0}/{1}'.format(opath, ofilename), tweets)


def main():

    path = 'cleaned_tweets/by_hashtag'
    opath = 'cleaned_tweets/merged'
    tweet_files = [tweet_file[:-5] for tweet_file in listdir(path) if not tweet_file.startswith(".")
                   and not tweet_file.endswith("_retweets_cleaned.json") ]

    for tweet_file in tweet_files:
        try:
            tweets = load_json("{0}/{1}".format(path, tweet_file))
            retweets = load_json("{0}/{1}_retweets_cleaned".format(path, tweet_file[0:-8]))
            print "processing {0}".format(tweet_file)

            merger(tweets, retweets, tweet_file, opath=opath)
        except:
            print "Exception"
            continue


if __name__ == '__main__':
    main()