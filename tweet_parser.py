__author__ = 'Pablo'


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from util import drop_none
from util import get_timestamps
from util import load_json
from util import save_json
from util import get_user_ids


class Scalor():
    """
    Scaling class for the input of KMeans
    """
    def __init__(self, data):
        self.data = data

    def scale(self, x):
        y = np.copy(x)

        y[:, 0] = np.log10(x[:, 0])
        y[:, 1] = x[:, 1]/np.max(self.data[:, 1])*2
        return y

    def unscale(self, x):
        y = np.copy(x)
        y[:, 0] = 10**(x[:,0])
        y[:, 1] = x[:, 1]*np.max(self.data[:, 1])/2
        return y


def retrieve_authors(tweets, my_dictionary, clipper=None):
    """
    Computes the author appeareances in the list of tweets
    :param tweets:
    :param my_dictionary:
    :param clipper:
    :return:
    """

    for i, tweet in enumerate(tweets):
        if tweet['user']['id'] not in my_dictionary.keys():
            my_dictionary[tweet['user']['id']] = 1
        else:
            my_dictionary[tweet['user']['id']] += 1
        if clipper is not None and i == clipper:
            break


def author_promiscuity(tweets, my_dictionary, clipper=None):
    """
    Computes the number if the author appears in the list of tweets and adds one to its entry in my_dictionary
    :param tweets:
    :param my_dictionary:
    :param clipper:
    :return:
    """
    visited = [None]*len(tweets)
    for i, tweet in enumerate(tweets):
        if tweet['user']['id'] not in visited:
            if tweet['user']['id'] not in my_dictionary.keys():
                my_dictionary[tweet['user']['id']] = 1
            else:
                my_dictionary[tweet['user']['id']] += 1
            visited[i] = tweet['user']['id']
        if clipper is not None and i == clipper:
            break


def parse_authors(files, path, filename='author_counts', filename_promiscuity='author_promiscuity', clipper=-1):
    """
    Computes the number of appearences of each author in files
    :param files:
    :param path:
    :param filename:
    :param filename_promiscuity:
    :param clipper:
    :return:
    """
    authors = dict()
    promiscuity = dict()
    for tweet_file in files:
        tweets = load_json('{0}/{1}'.format(path, tweet_file))
        retrieve_authors(tweets, authors, clipper=clipper)
        author_promiscuity(tweets, promiscuity, clipper=clipper)
        print "file {0} parsed".format(tweet_file)
    if clipper > 0:
        save_json('{0}_{1}'.format(filename, clipper), authors)
        save_json('{0}_{1}'.format(filename_promiscuity, clipper), promiscuity)
    else:
        save_json(filename, authors)
        save_json(filename_promiscuity, promiscuity)


def process_promiscuity(filename='author_promiscuity'):
    """
    Histogram of the promiscuity of the different users
    :param filename:
    :return:
    """
    promiscuity = load_json(filename)
    author_counts = np.zeros(len(promiscuity.keys()))
    author_names = [None] * len(promiscuity.keys())

    min_promiscuity = 5

    for i, user in enumerate(promiscuity.keys()):
        author_counts[i] = promiscuity[user]
        author_names[i] = user

    frequencies = np.sort(author_counts)[-1::-1]

    nbins = 100

    fig, ax = plt.subplots()
    ax.hist(frequencies[frequencies >= min_promiscuity], nbins)
    plt.show()


def process_authors(filename='authors'):
    """
    Takes a dictionary that has user id as a key and #tweets posted in total under all the hashtags.
    :param filename:
    :return:
    """

    authors = load_json(filename)
    author_counts = np.zeros(len(authors.keys()))
    author_names = [None] * len(authors.keys())

    for i, user in enumerate(authors.keys()):
        author_counts[i] = authors[user]
        author_names[i] = user

    frequencies = np.sort(author_counts)[-1::-1]
    x = np.arange(frequencies.size)

    fig, ax = plt.subplots(4,1)
    l_bound = [13, 50]
    nbins = 100
    xticks = np.array([0, 10, 20, 30, 40, 50, 80, 100, 200, 500, 1000, 1500, 200])
    xlim = [[0, 20], [l_bound[1], 300]]

    for i, l in enumerate(l_bound):
        ax[i].hist(frequencies[frequencies>l], nbins)
        ax[i].xaxis.set_ticks(xticks)
        ax[i].set_xlim(xlim[i])
        print np.sum(frequencies>l)
    plt.show()


def tweet_clustering(filenames=['author_counts_tweets_3000', 'author_promiscuity_tweets_3000'],
                            th=4, oth=45, n_clusters=5):
    """
    Performs K-means clustering and plots results
    :param filenames:
    :param th: minimum number of tweets
    :param th: maximum number of tweets
    :return:
    """
    x_file = load_json(filenames[0])
    y_file = load_json(filenames[1])

    data = np.zeros((len(x_file.keys()), 2))
    data_original_format = np.zeros((len(x_file.keys()), 2))

    for i, user in enumerate(x_file.keys()):
        data[i, 0] = x_file[user]
        data_original_format[i, 0] = x_file[user]

    for i, user in enumerate(y_file.keys()):
        data[i, 1] = x_file[user]/y_file[user]
        data_original_format[i, 1] = y_file[user]

    #Scale data
    scalor = Scalor(data)

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    no_outliers = np.logical_and(data[:,0] > th, data[:,0] < oth)

    data_no_outliers = data[no_outliers]
    if scalor is None:
        kmeans.fit(data_no_outliers)
        labels = kmeans.predict(data)
        unscaled_centroids = kmeans.cluster_centers_
    else:
        kmeans.fit(scalor.scale(data_no_outliers))
        labels = kmeans.predict(scalor.scale(data))
        unscaled_centroids = scalor.unscale(kmeans.cluster_centers_)


    colors = ['b', 'k', 'g', '#fdc086', '#8856a7', '#f7fcb9', '#c994c7', '#636363', '#ffeda0', '#c51b8a']

    print "Estimations"
    for i in xrange(np.unique(labels).size):
        print "# Soldiers rank {0} ({1}): {2}, {3} %, generated content: {4}, {5}%". format(i,
                                                                                            colors[i],
                                                                                      np.sum(labels==i),
                                                                                      np.sum(labels==i)*100./labels.size,
                                                                                      np.sum(data[labels==i, 0]),
                                                                                      np.sum(data[labels==i, 0])*100./np.sum(data[:,0]))

    print "Army size: {0}".format(labels.size)

    fig, ax = plt.subplots(2,2)
    ax = ax.flatten()
    for i in xrange(np.unique(labels).size):
        ax[0].scatter(np.log10(data[labels==i, 0]), data[labels==i, 1], color=colors[i])
    ax[0].scatter(np.log10(unscaled_centroids[:,0]), unscaled_centroids[:, 1], color='r')
    ax[0].set_xlabel('# tweets [Log scale]')
    ax[0].set_ylabel('tweets per hashtag')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].tick_params(top='off', right='off')

    for i in xrange(np.unique(labels).size):
        ax[1].scatter(np.log10(data_original_format[labels==i, 0]), data_original_format[labels==i, 1], color=colors[i])
    ax[1].scatter(np.log10(unscaled_centroids[:,0]), unscaled_centroids[:, 0]/unscaled_centroids[:, 1], color='r')

    ax[1].set_xlabel('# tweets [log scale]')
    ax[1].set_ylabel('promiscuity')
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].tick_params(top='off', right='off')

    if scalor is not None:
        scaled_data = scalor.scale(data_no_outliers)
        labels_no_outliers = labels[no_outliers]
        for i in xrange(np.unique(labels).size):
            ax[2].scatter(scaled_data[labels_no_outliers==i, 0], scaled_data[labels_no_outliers==i, 1], color=colors[i])

        ax[2].scatter(scalor.scale(unscaled_centroids)[:,0], scalor.scale(unscaled_centroids)[:, 1], color='r')

        ax[2].set_xlabel('# tweets [nat scale]')
        ax[2].set_ylabel('tweets per hashtag')
        ax[2].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['left'].set_visible(False)
        ax[2].spines['bottom'].set_visible(False)
        ax[2].tick_params(top='off', right='off')
        ax[2].axis('equal')


    for i in xrange(np.unique(labels).size):
        ax[3].scatter(data_original_format[labels==i, 0], data_original_format[labels==i, 1], color=colors[i])

    ax[3].scatter(unscaled_centroids[:,0], unscaled_centroids[:, 0]/unscaled_centroids[:, 1], color='r')
    ax[3].set_xlabel('# tweets [nat scale]')
    ax[3].set_ylabel('Promiscuity')
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['left'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[3].tick_params(top='off', right='off')
    plt.show()

    users = np.array(x_file.keys())
    propagandists = users[np.logical_or(labels==3, labels==4)]#np.logical_or.reduce((labels==2, labels==3, labels==4))]
    print propagandists.shape
    save_json("results/football/propagandists_{0}".format(filenames[0]), list(propagandists))


def tweet_retweet_ratio(files, path):
    """
    Barplot representing the ratio of tweet-retweet per hashtag.
    :param files: List where each element is another list with two elements: file containing tweets, file containing
    retweets
    :param path:
    :return:
    """

    #Initialize variables
    n_tweets = [None] * len(files)
    n_tweets_proportion = [None] * len(files)
    n_retweets = [None] * len(files)
    n_retweets_proportion = [None] * len(files)
    names = [None] * len(files)
    n_total = [None] * len(files)


    #Loop over the files to extract number of tweets and retweets per hashtag
    for i, hashtag_files in enumerate(files):

        tweets = load_json('{0}/{1}'.format(path, hashtag_files[0]))
        retweets = load_json('{0}/{1}'.format(path, hashtag_files[1]))
        n_tweets[i] = len(tweets)
        n_retweets[i] = len(retweets)
        n_total[i] = (n_tweets[i] + n_retweets[i])
        n_tweets_proportion[i] = float(n_tweets[i]) / n_total[i]
        n_retweets_proportion[i] = float(n_retweets[i]) / n_total[i]

        names[i] = hashtag_files[0]
        print "file {0} parsed".format(hashtag_files[0])

    row_format ="{:>15}\t" * 4

    #Print results in table
    print "Hashgtag {:<8} # Tweet {:<8} #Retweets {:<8} Total{:<8}"
    for i in xrange(len(names)):
        data = [names[i], n_tweets[i], n_retweets[i], n_total[i]]
        print row_format.format(*data)


    #Barplot colors
    colors = ['#b2df8a', '#a6cee3']
    #Barplot layout definition
    ind = np.arange(len(files))
    width = 0.35
    y_ticks = np.arange(0, 1.2, 0.2)

    #Barplot
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, n_tweets_proportion, width, color=colors[0], edgecolor='none')
    rects2 = ax.bar(ind + width, n_retweets_proportion, width, color=colors[1], edgecolor='none')
    for y in y_ticks[1:]:
        ax.plot([ind[0], ind[-1] + 1], [y, y], color='w')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(np.arange(len(files)))
    ax.legend( (rects1[0], rects2[0]), ('Tweets', 'Retweets'), frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(top='off', right='off', bottom='off', left='off')
    ax.set_ylim([0, 1.1])
    ax.set_xlabel('Hashtag number')
    ax.set_ylabel('Proportion')
    plt.show()



def hashtag_evolution(files, path, propagandist_file=None, windows=None):
    """
    plots the proportion of content generated by propagandists over time
    :param files:
    :param path:
    :param propagandist_file:
    :param clipper:
    :param windows:
    :return:
    """
    #fix, ax = plt.subplots(len(files), 1)
    #end = [60*20, 60*20, 1.7e4]
    #start = [7, 1, 0, 0, 0]

    if windows is None:
        windows = [None] * len(files)

    if propagandist_file is not None:
#        propagandists = np.genfromtxt(propagandist_file, delimiter=',')
         propagandists = np.array(load_json(propagandist_file), dtype=int)
    for i, hashtag_files in enumerate(files):

        print hashtag_files[0]

        tweets = load_json('{0}/{1}'.format(path, hashtag_files[0]))
        retweets = load_json('{0}/{1}'.format(path, hashtag_files[1]))

        tweet_timestamps = np.array(get_timestamps(tweets))
        retweet_timestamps = np.array(get_timestamps(retweets))

        all_timestamps = np.hstack((tweet_timestamps, retweet_timestamps))
        order = np.argsort(all_timestamps)
        all_timestamps = np.sort(all_timestamps)

        origin = np.min(all_timestamps)
        if windows[i] is not None:
            print origin + windows[i][0]
        else:
            print origin
        all_timestamps = (all_timestamps - origin)/60.
        tweet_timestamps = (tweet_timestamps - origin)/60.
        retweet_timestamps = (retweet_timestamps - origin)/60.

        indices = np.arange(order.size)

        print "Total number of tweets: {0}\nTotal Number of retweets: {1}".format(
            tweet_timestamps.size, retweet_timestamps.size)

        if propagandist_file is None:

            colors = ['r', 'b']

            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i, t in enumerate(all_timestamps):
                ax.scatter(t, i + 1, color=colors[order[i]<tweet_timestamps.size])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xlim([0, 120])
            plt.show()


        else:

            l_size = 20
            tweet_user_ids = get_user_ids(tweets)
            retweet_user_ids = get_user_ids(retweets)

            tweet_propagandistic_users = np.intersect1d(tweet_user_ids, propagandists)
            retweet_propagandistic_users = np.intersect1d(retweet_user_ids, propagandists)

            propagandistic_tweet_timestamps = tweet_timestamps[np.in1d(tweet_user_ids, tweet_propagandistic_users)]
            propagandistic_retweet_timestamps = retweet_timestamps[np.in1d(retweet_user_ids,
                                                                           retweet_propagandistic_users)]

            legit_tweet_timestamps = np.setdiff1d(tweet_timestamps, propagandistic_tweet_timestamps)
            legit_retweet_timestamps = np.setdiff1d(retweet_timestamps, propagandistic_retweet_timestamps)

            print "Estimated propagandistic tweets: {0}, equals to {1}%\nEstimated propandistic retweets: {2}, equals" \
                  " to {3}%".format(propagandistic_tweet_timestamps.size,
                                    propagandistic_tweet_timestamps.size * 100./ tweet_timestamps.size,
                                    propagandistic_retweet_timestamps.size,
                                    propagandistic_retweet_timestamps.size * 100. / retweet_timestamps.size)


            colors = [['#a6cee3', '#1f78b4'], ['#b2df8a', '#33a02c']]

            legit_tweet_cumsum = np.cumsum(np.logical_not(np.in1d(tweet_user_ids, tweet_propagandistic_users)))
            propagandistic_tweet_cumsum = np.cumsum(np.in1d(tweet_user_ids, tweet_propagandistic_users))

            legit_retweet_cumsum = np.cumsum(np.logical_not(np.in1d(retweet_user_ids, retweet_propagandistic_users)))
            propagandistic_retweet_cumsum = np.cumsum(np.in1d(retweet_user_ids, retweet_propagandistic_users))


            total_user_ids = np.hstack((tweet_user_ids, retweet_user_ids))[order]

            total_legit_cumsum = np.cumsum(np.logical_not(np.logical_or(np.in1d(total_user_ids,
                                                                                tweet_propagandistic_users),
                                                                        np.in1d(total_user_ids,
                                                                                retweet_propagandistic_users))))
            total_propagandistic_cumsum = np.cumsum(np.logical_or(np.in1d(total_user_ids, tweet_propagandistic_users),
                                                                  np.in1d(total_user_ids, retweet_propagandistic_users)))


            fig = plt.figure(figsize=(30,20))
            ax = fig.add_subplot(311)
            if windows[i] is None:
                y_values =  propagandistic_tweet_cumsum * 1./(propagandistic_tweet_cumsum + legit_tweet_cumsum)
                ax.scatter(tweet_timestamps, y_values, edgecolors='none')
            else:
                y_values =  propagandistic_tweet_cumsum * 1./(propagandistic_tweet_cumsum + legit_tweet_cumsum)
                ax.scatter(tweet_timestamps - windows[i][0], y_values, edgecolors='none')
                ax.set_xlim((0, windows[i][1] - windows[i][0]))

            if np.max(y_values) < 0.1:
                ax.set_ylim(-0.05, 0.11)

            tmp = ax.get_yticks()
            ax.set_yticks(np.arange(0, tmp[-1], 0.1))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_ylabel('Proportion', fontsize=l_size)
            ax.set_xlabel('Time (min)', fontsize=l_size)
            ax.set_title('Tweets', fontsize=l_size)
            ax.tick_params(top='off')
            ax.tick_params(axis='both', which='major', labelsize=l_size)

            ax = fig.add_subplot(312)
            if windows[i] is None:
                y_values = propagandistic_retweet_cumsum * 1./(propagandistic_retweet_cumsum + legit_retweet_cumsum)
                ax.scatter(retweet_timestamps, y_values, edgecolors='none')
            else:
                y_values = propagandistic_retweet_cumsum * 1./(propagandistic_retweet_cumsum + legit_retweet_cumsum)
                ax.scatter(retweet_timestamps - windows[i][0], y_values, edgecolors='none')
                ax.set_xlim((0, windows[i][1] - windows[i][0]))

            if np.max(y_values) < 0.1:
                ax.set_ylim(-0.05, 0.11)
            tmp = ax.get_yticks()
            ax.set_yticks(np.arange(0, tmp[-1], 0.1))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_ylabel('Proportion', fontsize=l_size)
            ax.set_xlabel('Time (min)', fontsize=l_size)
            ax.set_title('Retweets', fontsize=l_size)
            ax.tick_params(top='off')
            ax.tick_params(axis='both', which='major', labelsize=l_size)

            ax = fig.add_subplot(313)

            if windows[i] is None:
                y_values = total_propagandistic_cumsum * 1./(total_propagandistic_cumsum + total_legit_cumsum)
                ax.scatter(all_timestamps, y_values, edgecolors='none')
            else:
                y_values = total_propagandistic_cumsum * 1./(total_propagandistic_cumsum + total_legit_cumsum)
                ax.scatter(all_timestamps - windows[i][0], y_values, edgecolors='none')
                ax.set_xlim((0, windows[i][1] - windows[i][0]))

            if np.max(y_values) < 0.1:
                ax.set_ylim(-0.05, 0.11)
            tmp = ax.get_yticks()
            ax.set_yticks(np.arange(0, tmp[-1], 0.1))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(top='off')
            ax.set_ylabel('Proportion', fontsize=l_size)
            ax.set_xlabel('Time (min)', fontsize=l_size)
            ax.set_title('Total', fontsize=l_size)
            ax.tick_params(axis='both', which='major', labelsize=l_size)
            plt.savefig("results/football/evolution/{0}.png".format(hashtag_files[0]))


def main():
    pass

    #Some examples of use:
    #path = 'cleaned_tweets/by_hashtag'

    #parse_authors(files, path, filename='football_author_counts', filename_promiscuity='football_author_promiscuity', clipper=3000)
    #files = [['#CrisisBipartidismoM4_cleaned', '#CrisisBipartidismoM4_retweets_cleaned'],
    #      ['#ElCambioEmpiezaEnAndalucia_cleaned', '#ElCambioEmpiezaEnAndalucia_retweets_cleaned']]

    #process_promiscuity('football/football_author_promiscuity_3000')

    #tweet_clustering(filenames=['author_counts_tweets_3000', 'author_promiscuity_tweets_3000'])#filenames=['author_counts_tweets_3000', 'author_promiscuity_tweets_3000'])
    
    #Input for tweet_retweet_ratio and hashgtag_evolution
    #files = [[tweet_file[0:-5], '{0}_retweets_cleaned'.format(tweet_file[0:-13])] for tweet_file in listdir(path)
    #         if not tweet_file.startswith(".") and not tweet_file.endswith('_retweets_cleaned.json')]

    #Windows for hashtag_evolution
    #windows = [None, [0, 60], [12796, 12850], [230, 300], [0, 100], [820, 900], [343180, 343280], [21650, 22e3], [0, 100],
    #         [0, 100], [9800, 9900], [0, 1500], [0, 3000], [7700, 8200], [300, 800], [100, 600], [1000, 1500]]
    #windows = None
    #hashtag_evolution(files, path, propagandist_file='propagandists_football_author_counts_3000', windows=windows)
    #tweet_retweet_ratio(files, path)

if __name__ == '__main__':
    main()