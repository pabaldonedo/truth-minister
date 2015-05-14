from util import load_json
from util import save_json
from util import retrieve_files
from util import get_timestamps
from util import list_dir
from time import mktime
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def tweet_tree(path='cleaned_tweets/by_hashtag', prop_file='propagandists_merged_3000'):
    """
    Builds a dictionary containing user_id, screen_name, tweet_id, tweettime, topic and if the user is propagadandist
    for each tweet
    :param path:
    :param prop_file:
    :return:
    """
    files = retrieve_files(path)
    prop = load_json(prop_file)
    tree = dict()
    for f in files:
        if f.endswith('retweets_cleaned.json') or f =='#AsambleaCiudadana_cleaned.json':
            continue
        x = load_json('{0}/{1}'.format(path,f[:-5]))
        for i, tweet in enumerate(x):
            twitter = str(tweet['user']['id'])
            info = {'user_id': twitter,
                    'screen_name': tweet['user']['screen_name'],
                    'tweet_id': tweet['id'],
                    'tweet_time': tweet['created_at'],
                    'topic': f[:-13],
                    'propagandist': twitter in prop}
            if twitter in tree.keys():
                tree[twitter] = tree[twitter] + [info]
            else:
                tree[twitter] = [info]
            if (i+1) % 1000 == 0:
                print "{0} tweets processed in file {1}".format(i+1, f[:-5])
        print "file {0} processed".format(f)
    save_json('tweet_behaviour', tree)


def twitter_behaviour(filename='tweet_behaviour', time_file='tweets_origin_time'):
    """
    Computes the mean, standard deviation and median time of tweeting for each user taking the 0 time for each
    hashtag the one proviede in time_file
    :param filename:
    :param time_file: file containing the timestamp of the origin of each hashtag
    :return:
    """
    tweets = load_json(filename)
    propagandist_activity = dict()
    windows = load_json(time_file)
    too_fast = 0
    for j, k in enumerate(tweets.keys()):

        user_name = str(tweets[k][0]['screen_name'])
        if user_name not in propagandist_activity.keys():
            propagandist_activity[user_name] = dict()

        user_list = tweets[k]
        for i in xrange(len(user_list)):

            t_time = mktime(datetime.strptime(user_list[i]['tweet_time'], '%a %b %d %H:%M:%S +0000 %Y').timetuple())
            rt = user_list[i]
            t_time -= windows[rt['topic']]
            if t_time < 0:
                too_fast += 1
                continue
            if rt['topic'] in propagandist_activity[user_name].keys():
                propagandist_activity[user_name][rt['topic']] = \
                 min(t_time, propagandist_activity[user_name][rt['topic']])
             #   propagandist_activity[user_name][rt['topic']] += np.array([t_diff, t2_diff, 1])
            else:
                propagandist_activity[user_name][rt['topic']] = t_time
              #  propagandist_activity[user_name][rt['topic']] = np.array([t_diff, t2_diff, 1])
        if (j+1) % 1000 == 0:
            print '{0}/{1} parsed'.format(j+1, len(tweets.keys()))

    files = retrieve_files('cleaned_tweets/by_hashtag')
    columns = []
    for f in files:
        if not f.endswith('retweets_cleaned.json'):
            continue
        columns += ['{0}_min'.format(f[:-22])]

    columns += ['total_mean', 'total_std', 'total_median']

    df = pd.DataFrame(columns=columns)
    for j, k in enumerate(propagandist_activity.keys()):
        user_dict = propagandist_activity[k]
        total = []
        for topic in user_dict.keys():
            #mean = user_dict[topic][0]*1./user_dict[topic][2]/60.
            #var =  user_dict[topic][1]*1./user_dict[topic][2]/60**2 - mean**2
            df.loc[k, '{0}_min'.format(topic)] = user_dict[topic]/60.

            total += [user_dict[topic]]
        df.loc[k, 'total_mean'] = np.mean(np.array(total)/60.)
        df.loc[k, 'total_std'] = np.std(np.array(total)/60.)
        df.loc[k, 'total_median'] = np.median(np.array(total)/60.)
        if (j+1) % 1000 == 0:
            print '{0}/{1} parsed'.format(j+1, len(propagandist_activity.keys()))
    df = df.dropna(how='all')
    df.to_csv('results/tweets_times_all_people.csv')    


def propagandist_behaviour(filename='tweet_behaviour', prop_file='propagandists_merged_3000',
                                                                    time_file='tweets_origin_time'):
    """
    Computes the mean, standard deviation and median time of tweeting for each propagandistic user taking the 0 time for each
    hashtag the one proviede in time_file
    :param filename:
    :param time_file: file containing the timestamp of the origin of each hashtag
    :return:
    """
    tweets = load_json(filename)
    prop = load_json(prop_file)

    propagandist_activity = dict()
    windows = load_json(time_file)
    tweets_numbers = 0
    for k in tweets.keys():
        if str(k) not in prop:
            continue

        user_name = str(tweets[k][0]['screen_name'])
        if user_name not in propagandist_activity.keys():
            propagandist_activity[user_name] = dict()

        user_list = tweets[k]
        for i in xrange(len(user_list)):

            t_time = mktime(datetime.strptime(user_list[i]['tweet_time'], '%a %b %d %H:%M:%S +0000 %Y').timetuple())
            rt = user_list[i]

            t_time -= windows[rt['topic']]

            if t_time < 0:
                continue
            tweets_numbers += 1
            if rt['topic'] in propagandist_activity[user_name].keys():
                propagandist_activity[user_name][rt['topic']] =\
                                        min(t_time, propagandist_activity[user_name][rt['topic']])
            else:
                propagandist_activity[user_name][rt['topic']] = t_time

    files = retrieve_files('cleaned_tweets/by_hashtag')
    columns = []
    for f in files:
        if f.endswith('retweets_cleaned.json'):
            continue
        columns += ['{0}_min'.format(f[:-13])]

    columns += ['total_mean', 'total_std', 'total_median']

    df = pd.DataFrame(columns=columns)

    for k in propagandist_activity.keys():
        user_dict = propagandist_activity[k]
        total = []
        for topic in user_dict.keys():
            #mean = user_dict[topic][0]*1./user_dict[topic][2]/60.
            #var =  user_dict[topic][1]*1./user_dict[topic][2]/60**2 - mean**2
            df.loc[k, '{0}_min'.format(topic)] = user_dict[topic]/60.

            total += [user_dict[topic]]
        df.loc[k, 'total_mean'] = np.mean(np.array(total)/60.)
        df.loc[k, 'total_std'] = np.std(np.array(total)/60.)
        df.loc[k, 'total_median'] = np.median(np.array(total)/60.)

    print "Tweets: {0}".format(tweets_numbers)
    df.to_csv('results/tweets_times.csv')


def origin_tweet_time(windows, files, path):
    """
    generates a json file containing the timestamp used as 0 time for each hashtag.
    :param windows: valid timestamp window. Used for dropping tweet outliers (tweets outside the window).
    :param files:
    :param path:
    :return:
    """

    #windows = [None, [0, 60], [12796, 12850], [230, 300], [0, 100], [820, 900], [343180, 343280], [21650, 22e3], [0, 100],
    #           [0, 100], [9800, 9900], [0, 1500], [0, 3000], [7700, 8200], [300, 800], [100, 600], [1000, 1500]]
    #path = 'cleaned_tweets/by_hashtag'
#    files = [[tweet_file[0:-5], '{0}_retweets_cleaned'.format(tweet_file[0:-13])] for tweet_file in listdir(path)
#             if not tweet_file.startswith(".") and not tweet_file.endswith('_retweets_cleaned.json')]

    origin_times = dict()
    for i, hashtag_files in enumerate(files):
        tweets = load_json('{0}/{1}'.format(path, hashtag_files[0]))
        retweets = load_json('{0}/{1}'.format(path, hashtag_files[1]))

        tweet_timestamps = get_timestamps(tweets)
        retweet_timestamps = np.array(get_timestamps(retweets))

        all_timestamps = np.hstack((tweet_timestamps, retweet_timestamps))
        all_timestamps = np.sort(all_timestamps)

        origin = np.min(all_timestamps)
        if windows[i] is not None:
            origin_times[hashtag_files[0][:-8]] = tweet_timestamps[tweet_timestamps>=origin+windows[i][0]*60][0]
        else:
            origin_times[hashtag_files[0][:-8]] = origin
        print hashtag_files[0]

    save_json('tweets_origin_time', origin_times)


def plot_tweet(df):
    """
    Plots histogram with median time for tweeting of each user
    :param df:
    :return:
    """
    median = df['total_median'].values 
    double_axis = True
    upper_bound = 300
    h, bins = np.histogram(median[median <= upper_bound], bins=np.linspace(0, upper_bound + 1, 100))
    h = h*100./median.size
    cdf = np.cumsum(h)
    colors = ['#b2df8a', '#a6cee3']

    fig, ax = plt.subplots()
    ax.bar(bins[:-1], h, width=bins[1]-bins[0], color=colors[0], edgecolor='none')

    y_ticks = np.arange(0, 10, 1)

    for y in y_ticks[1:]:
        ax.plot([bins[0], bins[-1]], [y, y], color='w')

    ax.set_xlabel('median time for first tweet (min)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(top='off', right='off', bottom='off', left='off')
    ax.set_xlim([0, upper_bound])

    if double_axis is True:

        ax2 = ax.twinx()
        ax2.plot(bins[:-1] + (bins[1]-bins[0])/2., cdf, color=colors[1])
        #ax2.set_ylabel('% propagandists')
        #for tl in ax2.get_yticklabels():
        #    tl.set_color(colors[1])
        ax2.plot([bins[0], bins[-1]], [50, 50], 'r--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params(top='off', right='off', bottom='off', left='off')
        ax2.set_xlim([0, upper_bound])

    else:
        ax.plot(bins[:-1] + (bins[1]-bins[0])/2., cdf, color=colors[1])

    ax.set_ylabel('% users')
    ax.set_ylim([0,10])
    plt.show()
