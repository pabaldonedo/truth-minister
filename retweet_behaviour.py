from util import load_json
from util import save_json
from util import retrieve_files
from time import mktime
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def retweet_tree(path='cleaned_tweets/by_hashtag', prop_file='propagandists_merged_3000'):
    """
    Builds a dictionary for each tweet containing user_id of the original tweet, his screen_name, tweet_id, tweet time, topic,
    user_id of the user retweeting the tweet, his screen_name, id of the retweet, its  time and if the user retweeting
    is propagadandist
    :param path:
    :param prop_file:
    :return:
    """
    files = retrieve_files(path)
    prop = load_json(prop_file)
    tree = dict()
    for f in files:
        if not f.endswith('retweets_cleaned.json'):
            continue
        x = load_json('{0}/{1}'.format(path,f[:-5]))
        for i, tweet in enumerate(x):
            retwitter = str(tweet['user']['id'])
            info = {'original_user_id': tweet['retweeted_status']['user']['id'],
                    'original_screen_name': tweet['retweeted_status']['user']['screen_name'],
                    'original_tweet_id': tweet['retweeted_status']['id'],
                    'original_tweet_time': tweet['retweeted_status']['created_at'],
                    'retweet_id': tweet['id'],
                    'retwitter_id': retwitter,
                    'retwitter_screen_name': str(tweet['user']['screen_name']),
                    'retweet_time': tweet['created_at'],
                    'topic': f[:-22],
                    'propagandist': retwitter in prop}
            if retwitter in tree.keys():
                tree[retwitter] = tree[retwitter] + [info]
            else:
                tree[retwitter] = [info]
            if (i+1) % 1000 == 0:
                print "{0} tweets processed in file {1}".format(i+1, f[:-5])
        print "file {0} processed".format(f)
    save_json('retweet_behaviour', tree)


def generate_spreadsheet(filename='retweet_behaviour', nodes='results/graphs/prop_retweets_nodes.csv',
                                                        edges='results/graphs/prop_retweets_edges.csv',
                                                        prop_file='propagandists_merged_3000',
                                                        only_prop=True):
    """
    Takes the file generated by retweet_behaviour and builds the input file for Gephi for a graph showing who are
    retweeting the propagandists.
    :param filename: input file
    :param nodes: output file containign the nodes of the graph
    :param edges: output file containing the edges of the graph
    :param prop_file: ids of propagandists
    :param only_prop:
    :return:
    """

    prop = load_json(prop_file)
    tree_data = load_json(filename)
    nodes_list = dict()
    with open(edges, 'w') as f_edge:
        f_edge.write('Source,Target,Type\n'.encode('utf8'))
        for j, x in enumerate(tree_data.keys()):
            if only_prop:
                if str(x) not in prop:
                    continue
            retweets = tree_data[x]
            nodes_list[retweets[0]['retwitter_screen_name'].encode('utf8')] = [str(retweets[0]['retwitter_id']).encode('utf8'), retweets[0]['retwitter_screen_name'].encode('utf8'),retweets[0]['propagandist']]


            for i in xrange(len(retweets)):
                nodes_list[retweets[i]['original_screen_name'].encode('utf8')] = [str(retweets[i]['original_user_id']).encode('utf8'), retweets[i]['original_screen_name'].encode('utf8'), str(retweets[i]['original_user_id']) in prop]
                f_edge.write('{0},{1},Directed\n'.format(str(retweets[i]['retwitter_id']).encode('utf8'),
                                                        str(retweets[i]['original_user_id']).encode('utf8')))
            if (j+1)%100 == 0:
                print "{0}/{1} users processed".format(j, len(tree_data.keys()))

    with open(nodes, 'w') as f_node:
        f_node.write('Id,Label,propagandist\n'.encode('utf8'))
        for key in nodes_list.keys():
            n = nodes_list[key]
            f_node.write('{0},{1},{2}\n'.format(n[0], n[1], n[2]))


def retwitter_behaviour(filename='retweet_behaviour'):
    """
    Computes the mean, standard deviation and median time of retweeting for each user since the moment tweet was created
    :param filename:
    :param time_file: file containing the timestamp of the origin of each hashtag
    :return:
    """
    retweets = load_json(filename)
    propagandist_activity = dict()

    for j, k in enumerate(retweets.keys()):

        user_name = str(retweets[k][0]['retwitter_screen_name'])
        if user_name not in propagandist_activity.keys():
            propagandist_activity[user_name] = dict()

        user_list = retweets[k]
        for i in xrange(len(user_list)):

            rt_time = mktime(datetime.strptime(user_list[i]['retweet_time'], '%a %b %d %H:%M:%S +0000 %Y').timetuple())
            tw_time = mktime(datetime.strptime(user_list[i]['original_tweet_time'], '%a %b %d %H:%M:%S +0000 %Y').timetuple())
            t_diff = rt_time - tw_time
            rt = user_list[i]
            if rt['topic'] in propagandist_activity[user_name].keys():
                propagandist_activity[user_name][rt['topic']] += [t_diff]
             #   propagandist_activity[user_name][rt['topic']] += np.array([t_diff, t2_diff, 1])
            else:
                propagandist_activity[user_name][rt['topic']] = [t_diff]
              #  propagandist_activity[user_name][rt['topic']] = np.array([t_diff, t2_diff, 1])
        if (j+1) % 1000 == 0:
            print '{0}/{1} parsed'.format(j+1, len(retweets.keys()))

    files = retrieve_files('cleaned_tweets/by_hashtag')
    columns = []
    for f in files:
        if not f.endswith('retweets_cleaned.json'):
            continue
        columns += ['{0}_mean'.format(f[:-22]), '{0}_std'.format(f[:-22]), '{0}_median'.format(f[:-22])]

    columns += ['total_mean', 'total_std', 'total_median']

    df = pd.DataFrame(columns=columns)
    for j, k in enumerate(propagandist_activity.keys()):
        user_dict = propagandist_activity[k]
        total = []
        for topic in user_dict.keys():
            #mean = user_dict[topic][0]*1./user_dict[topic][2]/60.
            #var =  user_dict[topic][1]*1./user_dict[topic][2]/60**2 - mean**2
            df.loc[k, '{0}_mean'.format(topic)] = np.mean(np.array(user_dict[topic])/60.)
            df.loc[k, '{0}_std'.format(topic)] = np.std(np.array(user_dict[topic])/60.)
            df.loc[k, '{0}_median'.format(topic)] = np.median(np.array(user_dict[topic])/60.)
            total += user_dict[topic]
        df.loc[k, 'total_mean'] = np.mean(np.array(total)/60.)
        df.loc[k, 'total_std'] = np.std(np.array(total)/60.)
        df.loc[k, 'total_median'] = np.median(np.array(total)/60.)
        if (j+1) % 1000 == 0:
            print '{0}/{1} parsed'.format(j+1, len(propagandist_activity.keys()))

    df.to_csv('results/retweets_times_all_people.csv')    



def propagandist_behaviour(filename='retweet_behaviour', prop_file='propagandists_merged_3000'):
    """
    Computes the mean, standard deviation and median time of retweeting for each propagandist since the moment tweet was created
    :param filename:
    :param time_file: file containing the timestamp of the origin of each hashtag
    :return:
    """
    retweets = load_json(filename)
    prop = load_json(prop_file)
    propagandist_activity = dict()

    for k in retweets.keys():
        if str(k) not in prop:
            continue

        user_name = str(retweets[k][0]['retwitter_screen_name'])
        if user_name not in propagandist_activity.keys():
            propagandist_activity[user_name] = dict()

        user_list = retweets[k]
        for i in xrange(len(user_list)):

            rt_time = mktime(datetime.strptime(user_list[i]['retweet_time'], '%a %b %d %H:%M:%S +0000 %Y').timetuple())
            tw_time = mktime(datetime.strptime(user_list[i]['original_tweet_time'], '%a %b %d %H:%M:%S +0000 %Y').timetuple())
            t_diff = rt_time - tw_time
            #t2_diff = (rt_time - tw_time)**2
            rt = user_list[i]
            if rt['topic'] in propagandist_activity[user_name].keys():
                propagandist_activity[user_name][rt['topic']] += [t_diff]
             #   propagandist_activity[user_name][rt['topic']] += np.array([t_diff, t2_diff, 1])
            else:
                propagandist_activity[user_name][rt['topic']] = [t_diff]
              #  propagandist_activity[user_name][rt['topic']] = np.array([t_diff, t2_diff, 1])

    files = retrieve_files('cleaned_tweets/by_hashtag')
    columns = []
    for f in files:
        if not f.endswith('retweets_cleaned.json'):
            continue
        columns += ['{0}_mean'.format(f[:-22]), '{0}_std'.format(f[:-22]), '{0}_median'.format(f[:-22])]

    columns += ['total_mean', 'total_std', 'total_median']

    df = pd.DataFrame(columns=columns)
    for k in propagandist_activity.keys():
        user_dict = propagandist_activity[k]
        total = []
        for topic in user_dict.keys():
            #mean = user_dict[topic][0]*1./user_dict[topic][2]/60.
            #var =  user_dict[topic][1]*1./user_dict[topic][2]/60**2 - mean**2
            df.loc[k, '{0}_mean'.format(topic)] = np.mean(np.array(user_dict[topic])/60.)
            df.loc[k, '{0}_std'.format(topic)] = np.std(np.array(user_dict[topic])/60.)
            df.loc[k, '{0}_median'.format(topic)] = np.median(np.array(user_dict[topic])/60.)
            total += user_dict[topic]
        df.loc[k, 'total_mean'] = np.mean(np.array(total)/60.)
        df.loc[k, 'total_std'] = np.std(np.array(total)/60.)
        df.loc[k, 'total_median'] = np.median(np.array(total)/60.)

    df.to_csv('results/retweets_times.csv')


def plot_retweets(df):
    """
    Plots histogram with median time for retweeting of each user
    :param df:
    :return:
    """
    median = df['total_median'].values 
    double_axis = True
    upper_bound = 30
    h, bins = np.histogram(median[median < upper_bound], bins=np.linspace(0, upper_bound + 1, 50))
    h = h*100./median.size
    cdf = np.cumsum(h)
    colors = ['#b2df8a', '#a6cee3']

    fig, ax = plt.subplots()
    ax.bar(bins[:-1], h, width=bins[1]-bins[0], color=colors[0], edgecolor='none')

    y_ticks = np.arange(0, 10, 1)

    for y in y_ticks[1:]:
        ax.plot([bins[0], bins[-1]], [y, y], color='w')

    ax.set_xlabel('median time for retweet (min)')
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


def main():
    #retweet_tree()
    generate_spreadsheet()

if __name__ == '__main__':
    main()