import string
import json

from nltk.tokenize import word_tokenize
from nltk.stem import wordnet
from nltk.tag import pos_tag
from nltk.corpus import stopwords, twitter_samples
from nltk import FreqDist

from collections import Counter
import matplotlib.pyplot as plt


# tweets = json.loads(open("eurovision_10000.json").read())

# tweets = twitter_samples.strings("tweets.20150430-223406.json")[:5000]
tweets = twitter_samples.strings("positive_tweets.json")[:5000]
# tweets = twitter_samples.strings("negative_tweets.json")[:1000]

stop_words = stopwords.words("english")
lemmatizer = wordnet.WordNetLemmatizer()

words = []
for tweet in tweets:
    tokens = word_tokenize(tweet)
    tagged = pos_tag(tokens)

    for token, tag in tagged:
        if tag.startswith("NN"):
            pos = "n"
        elif tag.startswith("VB"):
            pos = "v"
        else:
            pos = "a"
        lemma = lemmatizer.lemmatize(token, pos)
        if not lemma in stop_words:
            words.append(lemma)


emotions = json.loads(open("emotions.json").read())

emotions_list = []
for word in words:
    if word.lower() in emotions:
        emotions_list.append(emotions[word.lower()])


# frequencies = FreqDist(emotions_list)
# frequencies.plot(cumulative=False)


counted = Counter(emotions_list)

figure, axis = plt.subplots()
axis.bar(counted.keys(), counted.values())
figure.autofmt_xdate()

plt.show()
