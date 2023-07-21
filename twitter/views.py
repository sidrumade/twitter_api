from django.http import HttpResponse
import os
import pickle
import datetime
import time
import re
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
import tweepy
import nltk
import numpy as np
from deep_translator import GoogleTranslator
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
from nltk.corpus import stopwords
from .config import twitter_config

stop_words = set(stopwords.words('english'))
def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load('en_core_web_sm')  # 1
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()



from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
# Create your views here.



classifier = TextClassifier.load('en-sentiment')



def test(request):
    return HttpResponse("<h1>Hello</h1>")

@api_view(['GET'])
def read_root(request):
    return Response({"Hello": "World"},status=status.HTTP_200_OK)

@api_view(['GET'])
def info(request,user_name):
    obj=UserLookup(user_name)
    result = obj.get()
    return Response(result,status=status.HTTP_200_OK)

@api_view(['GET'])
def following(request,user_id):
    print('user_id p1===',user_id)
    obj=FollowLookup(user_id)
    result = obj.get()

    return Response(result,status=status.HTTP_200_OK)


@api_view(['GET'])
def tweets(request,user_name):
    print('user_name===',user_name)
    obj=TwittsLookup(user_name)
    result = obj.get()
    return Response(result,status=status.HTTP_200_OK)


@api_view(['GET'])
def wordcloud(request,user_name:str):
    obj=WordCloud(user_name)
    result = obj.get()
    return Response(result,status=status.HTTP_200_OK)


@api_view(['GET'])
def sentiments(request,user_name:str):
    obj=Sentiments(user_name)
    result = obj.get()
    return Response(result,status=status.HTTP_200_OK)




class TwittsLookup:
    def __init__(self, user_name , count=200):
        self.user_name = user_name  # ICICIBank
        self.count = count
        self.file_name = None

    def getFileName(self):
        return self.file_name

    def check_queue(self, name):
        while f'queue_{name}_{self.user_name}' in os.listdir('queue'):
            print(f'{self.user_name} waiting in queue')
            time.sleep(5)
        return True

    def create_queue(self, name):
        f = open(f'queue/queue_{name}_{self.user_name}', 'w')
        f.close()

    def check_exist(self, path):
        for item in os.listdir(path):
            if str(self.user_name) in item:
                self.file_name = item
                p_file = self.file_name.split('_')[1]
                available_date = datetime.datetime.strptime(p_file, '%Y-%m-%d %H:%M:%S.%f')
                print('file date', available_date)
                if (available_date + datetime.timedelta(
                        hours=3)) < datetime.datetime.now():  # data abailable but over 3 hours
                    print('file not exist returning false')
                    return False
                else:
                    print('file available under 3 hrs : returning true')
                    return True

    def destroy_queue(self, name):
        try:
            os.remove(rf'queue/queue_{name}_{self.user_name}')  # remove queue.
            print(f'removed queue: queue/queue_{self.user_name}')
        except Exception as e:
            print(f'Error in removing {name} queue')

    def connect_to_api(self):
        consumer_key = twitter_config['consumer_key']
        consumer_secret = twitter_config['consumer_secret']
        access_token = twitter_config['access_token']
        access_token_secret = twitter_config['access_token_secret']

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        return api


    def get_twitts(self):
        self.create_queue(name='tweets')
        api = self.connect_to_api()
        tweet_count = self.count  # Number of tweets to retrieve

        print('************************************************',tweet_count)

        json_response = []
        last_id = -1
        while len(json_response) < tweet_count:
            count = tweet_count - len(json_response)
            try:
                new_tweets = api.search_tweets(q='@' + self.user_name , count=count, max_id=str(last_id - 1))
                if not new_tweets:
                    break
                json_response.extend(new_tweets)
                last_id = new_tweets[-1].id
            except Exception as e:
                # Handle errors, such as rate limit exceeded
                print(f"Error: {str(e)}")
                break

        # delete old file
        if self.file_name:
            try:
                print('we have old tweets files so trying to remove it')
                os.remove(f'saved/{self.file_name}')  # old file exist, remove it.
            except Exception as e:
                print('error wile removing old file', e)
            print(f'after remove: saved/{self.file_name}', os.listdir())

        # write pickle /  save data
        json_response_list = [item._json for item in json_response]

        with open(f'saved/{self.user_name}_{datetime.datetime.now()}_.pkl', 'wb') as f:
            pickle.dump(json_response_list, f)

        self.destroy_queue(name="tweets")

        return json_response_list

    def get(self):
        if self.check_queue(name='tweets'):  # check if process is running.
            if self.check_exist('saved/'):  # check if tweets file is exists
                if self.file_name:
                    print(f'reading file saved/{self.file_name}')
                    with open(f'saved/{self.file_name}', 'rb') as f:  # read existing data
                        print('file exist-----')
                        data__ = pickle.load(f)
            else:

                print('file not exist feetching new tweets---')
                data__ = self.get_twitts()


        else:
            data__ = {}


        return data__




class UserLookup:
    def __init__(self, identity, by='username'):
        self.identity = identity
        self.by = by

    def connect_to_api(self):

        consumer_key = twitter_config['consumer_key']
        consumer_secret = twitter_config['consumer_secret']
        access_token = twitter_config['access_token']
        access_token_secret = twitter_config['access_token_secret']

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        return api

    def get(self):
        api = self.connect_to_api()
        try:
            if self.by == 'username':
                user = api.get_user(screen_name=self.identity)
            elif self.by == 'id':
                user = api.get_user(user_id=self.identity)

            
            return user._json
        except Exception:
            raise ValueError("Invalid value for 'by' parameter. Must be 'username' or 'id'.")




# following lookup
class FollowLookup:
    def __init__(self, user_id):
        print('user_id====',user_id,type(user_id))
        self.user_id = user_id
        self.consumer_key = twitter_config['consumer_key']
        self.consumer_secret = twitter_config['consumer_secret']
        self.access_token = twitter_config['access_token']
        self.access_token_secret = twitter_config['access_token_secret']

    def create_api(self):
        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        api = tweepy.API(auth)
        return api

    def get_following(self):
        api = self.create_api()
        following = api.get_friends(user_id=self.user_id, tweet_mode="extended")
        following_dict_list = [item._json for item in following]
        return following_dict_list

    def get(self):
        data__ = self.get_following()
        return data__



# swntiments
class Sentiments(TwittsLookup):
    def __init__(self, user_name):
        self.user_name = user_name
        self.file_name = None

    def get_sentiment(self, tweet):
        text_ = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", tweet.lower())
        sentence = Sentence(text_)
        classifier.predict(sentence)
        # return str(sentence.labels[0]) #.split()[0]
        if (len(sentence.labels) == 0): return 'Unclassified'
        return str(sentence.labels[0]).split(" ")[-2]

    def get(self):
        super().__init__(self.user_name)
        data__ = super().get()
        tweets_list = data__
        df = pd.DataFrame(tweets_list)
        print('inside sentiment')

        # check if tweets exists then read pickle of sentiments
        if super().check_queue(name='tweets'):  # check if tweets is in fetching state
            if super().check_queue(name='sentiments'):  # check if sentiments is in processing state
                if super().check_exist('saved/sentiments/') and (super().getFileName() is not None):
                    self.file_name = super().getFileName()
                    print(f'reading file saved/sentiments/{self.file_name}')
                    with open(f'saved/sentiments/{self.file_name}', 'rb') as f:  # read existing data
                        json_dic = pickle.load(f)

                else:
                    # create sentiment processing queue
                    super().create_queue(name='sentiments')
                    df['sentiment'] = df['text'].apply(lambda x: self.get_sentiment(x))
                    df = df[['text', 'sentiment']]
                    json_dic = df.groupby(by=['sentiment']).count().to_dict()

                    # delete old file
                    if self.file_name:
                        try:
                            print('removing old sentiments file', self.file_name)
                            os.remove(f'saved/sentiments/{self.file_name}')  # remove old sentiments
                        except Exception as e:
                            print('error wile removing old sentiment file', e)

                    # save pickle
                    print(f'saving----sentiments saved/sentiments/{self.user_name}_{datetime.datetime.now()}_.pkl')
                    with open(f'saved/sentiments/{self.user_name}_{datetime.datetime.now()}_.pkl', 'wb') as f:
                        pickle.dump(json_dic, f)
                    super().destroy_queue(name='sentiments')

        else:
            json_dic = {}

        return json_dic



def clean_tweet(tweet,extra = False):
    if type(tweet) == np.float64:
        return ""
    if not extra:
        temp = tweet.lower()
        temp = re.sub("'", "", temp) # to avoid removing contractions in english
        temp = re.sub("@[A-Za-z0-9_]+","", temp)
        temp = re.sub("#[A-Za-z0-9_]+","", temp)
        temp = re.sub(r'http\S+', '', temp)
        temp = re.sub('[()!?]', ' ', temp)
        temp = re.sub('\[.*?\]',' ', temp)
        temp = re.sub("[^a-z0-9]"," ", temp)
    else:
        tokens = re.split('\W+', tweet)
        temp = [ps.stem(w) for w in tokens if not w in stop_words]
        temp = [wn.lemmatize(word) for word in temp]  # lemmatizer
        temp = " ".join(word for word in temp)
    return temp


def translate_text(text_content):
    doc = nlp(text_content)  # 3
    detect_language = doc._.language
    if 'language' in detect_language and detect_language.get('language') != 'en':
        translated_text = GoogleTranslator(source='auto', target='en').translate(text_content)
    else:
        translated_text = text_content
    return translated_text

# WordCloud lookup
class WordCloud:
    def __init__(self, user_id):
        self.user_id = user_id  # 155659213


    def get(self):
        obj = TwittsLookup(self.user_id)
        data__ = obj.get()
        if len(data__) ==0 :
            return {'data': {} }

        df = pd.DataFrame(data__)
        df['clean_text'] = df['text'].map(lambda txt: clean_tweet(str(txt)))
        df['translated_text'] = df['clean_text'].map(lambda txt: translate_text(str(txt)))
        df['translated_text'] = df['translated_text'].map(lambda txt: clean_tweet(str(txt), extra=True))



        d = df.translated_text.str.split(expand=True).stack().value_counts()
        # final_word_dict = {k: v for k, v in d.to_dict().items() if len(k) > 3 and int(v) > 5 }
        final_word_list = [{'text': k , 'value': v } for k, v in d.to_dict().items() if len(k) > 3 and int(v) > 5 ]

        return {'data': final_word_list }


