import urllib
from util import load_json


def main():

    x = load_json('propagandists_info_3000')
    for i in x:
        image = i['profile_image_url_https']
        image_original = image.replace('_normal', '')
        urllib.urlretrieve(image_original, 'results/propagandists_photos/original/{0}.png'.format(i['screen_name']))
        urllib.urlretrieve(image, 'results/propagandists_photos/normal/{0}.png'.format(i['screen_name']))

