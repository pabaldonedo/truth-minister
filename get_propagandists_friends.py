from util import load_json
from util import save_json
from util import get_auth
import twitter
from robust_request import make_twitter_request


def main():
    filename =  'propagandists_info_3000' #sys.argv[1]#'topsy_results/#NoOlvidamosPPSOE_topsy'
    propagandists = load_json(filename)
    output_filename = 'propagandists_friend_graph'#sys.argv[2]#'tweets/#NoOlvidamosPPSOE'

    user = 'juan'
    print "user: {0}".format(user)
   
    CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET = get_auth(user)

    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)

    twitter_api = twitter.Twitter(auth=auth)

    friends = dict()

    errors = 0
    for i, p in enumerate(propagandists):
        try:
            friends[p['id']] = make_twitter_request(twitter_api.friends.ids, user_id=str(p['id']))
        except:
            print "ERROR at propagandist number {0} with id {1}".format(i, p['id'])
            errors += 1
    save_json('propagandists_friends', friends)


if __name__ == '__main__':
    main()
