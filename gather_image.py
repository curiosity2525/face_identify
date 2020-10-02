import numpy
import cv2
from requests import exceptions
import argparse
import requests
import os

API_KEY = "APIキーを入力" #APIキーを記入
MAX_RESULTS = 100
GROUP_SIZE = 50

#取得したエンドポイントURL
URL = "取得したエンドポイントURL" #URLを記入
OUTPUT = '/OUTPUT先のパス' #パスを記入

if not os.path.isdir(OUTPUT):
    os.mkdir(OUTPUT)

EXCEPTIONS = set([IOError, FileNotFoundError,
    exceptions.RequestException, exceptions.HTTPError,
    exceptions.ConnectionError, exceptions.Timeout])

term = 'いきものがかり　吉岡聖恵' #任意に変更
headers = {"Ocp-Apim-Subscription-Key" : API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE, "imageType":"Photo", "color":"ColorOnly"}

# make the search
print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

# grab the results from the search, including the total number of
# estimated results returned by the Bing API
results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults,
    term))

# initialize the total number of images downloaded thus far
total = 0

# loop over the estimated number of results in `GROUP_SIZE` groups
for offset in range(0, estNumResults, GROUP_SIZE):
    # update the search parameters using the current offset, then
    # make the request to fetch the results
    print("[INFO] making request for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))
    # loop over the results
    for v in results["value"]:
        # try to download the image
        try:
            # make a request to download the image
            print("[INFO] fetching: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)

            # build the path to the output image
            print
            ext = v["contentUrl"][v["contentUrl"].rfind("."):v["contentUrl"].rfind("?") if v["contentUrl"].rfind("?") > 0 else None]
            p = os.path.sep.join([OUTPUT, "{}{}".format(
                str(total).zfill(8), ext)])

            # write the image to disk
            f = open(p, "wb")
            f.write(r.content)
            f.close()

        # catch any errors that would not unable us to download the
        # image
        except Exception as e:
            # check to see if our exception is in our list of
            # exceptions to check for
            if type(e) in EXCEPTIONS:
                print("[INFO] skipping: {}".format(v["contentUrl"]))
                continue
        # try to load the image from disk
        image = cv2.imread(p)

        # if the image is `None` then we could not properly load the
        # image from disk (so it should be ignored)
        if image is None:
            print("[INFO] deleting: {}".format(p))
            os.remove(p)
            continue

        # update the counter
        total += 1




