from util import load_json
from util import save_json


def generate_dict():
    """
    Generates a dictioanry containing the name and number of tweets and retweets of each propagandist
    :return:
    """
    x = load_json('propagandists_info_3000')
    propagandists = load_json('propagandists_merged_3000')
    tweets = load_json('author_counts_tweets_3000')
    retweets = load_json('author_counts_retweets_3000')
    result = dict()
    info = dict()

    for i in x:
        info[str(i['id'])] = i
     
    for p in propagandists:
        user_dict = {'name': info[str(p)]['screen_name']}
        total = 0
        if str(p) in tweets.keys():
            user_dict['tweets'] = tweets[str(p)]
            total += tweets[str(p)]
        else:
            user_dict['tweets'] = 0

        if str(p) in retweets.keys():
            user_dict['retweets'] = retweets[str(p)]
            total += retweets[str(p)]
        else:
            user_dict['retweets'] = 0

        user_dict['total'] = total
        result[str(p)] = user_dict

    save_json('propagandists_activity', result)


def generate_text(filename='propagandists_activity', ofilename='results/propagandists_activity.txt'):
    """
    From the dictionary of generate_dict outputs its information into a file
    :param filename:
    :param ofilename:
    :return:
    """
    activity = load_json(filename)
    with open(ofilename, 'w') as f:
        f.write('Name\t Tweets \t Retweets\t Total \t % retweets \n')
        for k in activity.keys():
            propagandist = activity[k]
            f.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(propagandist['name'], propagandist['tweets'],
                                                propagandist['retweets'], propagandist['total'],
                                                propagandist['retweets']*1./propagandist['total']))


def main():

   # generate_dict()
    generate_text()


if __name__ == '__main__':
    main()