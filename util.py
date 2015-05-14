__author__ = 'Pablo'

import numpy as np
from time import mktime
from datetime import datetime
from os import listdir
import io
import json
import pymongo


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


def load_json(filename):
    with io.open('{0}.json'.format(filename),
                 encoding='utf-8') as f:
        return json.load(f)


def save_json(filename, data):
    with io.open('{0}.json'.format(filename),
                 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(data, ensure_ascii=False)))


def drop_none(tweets):
    """
    Takes a list of tweets and removes all None entries.
    :param tweets: list of tweets.
    :return: list of tweets without None values.
    """
    return [tweet for tweet in tweets if tweet is not None]


def get_user_ids(tweets):
    """
    Takes a list of tweets and returns a list with their user_ids.
    :param tweets: list of tweets.
    :return: list with ids.
    """
    ids = np.zeros(len(tweets))
    for i, tweet in enumerate(tweets):
        ids[i] = tweet['user']['id']

    return ids


def get_timestamps(tweets):
    """
    Takes a list of tweets and returns a list with their timestamps.
    :param tweets: list of tweets.
    :return: list with timestamps.
    """
    timestamps = np.zeros(len(tweets))
    for i, tweet in enumerate(tweets):
        timestamps[i] = mktime(datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y').timetuple())

    return timestamps


def tweet_preprocesser(tweets, clipper=-1):
    """
    Takes a list of tweets, drop Nones, sort by timestamp and returns the first clipper tweets.
    :param tweets: list of tweets.
    :param clipper: Number of tweets to return. If 0 or negative values all tweets are returned.
    :return: ordered list of tweets.
    """
    tweets = drop_none(tweets)
    timestamps = get_timestamps(tweets)

    indices = np.argsort(timestamps)
    if clipper > 0:
        ordered_tweets = [None]*clipper
    else:
        ordered_tweets = [None]*len(tweets)

    for k, idx in enumerate(indices):
        ordered_tweets[k] = tweets[idx]
        if k == clipper - 1 and clipper > 0:
            break

    return ordered_tweets


def retrieve_files(path):
    return [tweet_file for tweet_file in listdir(path) if not tweet_file.startswith(".")]


def save_to_mongo(data, mongo_db, mongo_db_coll, **mongo_conn_kw):

    # Connects to the MongoDB server running on
    # localhost:27017 by default

    client = pymongo.MongoClient(**mongo_conn_kw)

    # Get a reference to a particular database

    db = client[mongo_db]

    # Reference a particular collection in the database

    coll = db[mongo_db_coll]

    # Perform a bulk insert and  return the IDs

    return coll.insert(data)


def load_from_mongo(mongo_db, mongo_db_coll, return_cursor=False,
                    criteria=None, projection=None, **mongo_conn_kw):

    # Optionally, use criteria and projection to limit the data that is
    # returned as documented in
    # http://docs.mongodb.org/manual/reference/method/db.collection.find/

    # Consider leveraging MongoDB's aggregations framework for more
    # sophisticated queries.

    client = pymongo.MongoClient(**mongo_conn_kw)
    db = client[mongo_db]
    coll = db[mongo_db_coll]

    if criteria is None:
        criteria = {}

    if projection is None:
        cursor = coll.find(criteria)
    else:
        cursor = coll.find(criteria, projection)

    # Returning a cursor is recommended for large amounts of data

    if return_cursor:
        return cursor
    else:
        return [ item for item in cursor ]