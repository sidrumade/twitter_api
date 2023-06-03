from copyreg import constructor
from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import pickle
import datetime

import requests
import re
import spacy
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
import time
nlp = spacy.load("en_core_web_sm")
classifier = TextClassifier.load('en-sentiment')

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




bearer_token = "AAAAAAAAAAAAAAAAAAAAAD0%2BcwEAAAAAOWA%2BPgz3FfGnOtse20Z0AGuepOk%3DDTdezCyLRQwlpjSuqu4hcJe5Bqrv6532EyhCGD8lHI11E3t7Ib"#os.environ.get("BEARER_TOKEN")


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserLookupPython"
    return r

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/info/{user_name}")
def info(user_name:str):
    obj=UserLookup(user_name)
    result = obj.get()
    return result

@app.get("/following/{user_id}")
def following(user_id:int):
    obj=FollowLookup(user_id)
    result = obj.get()
    return result

@app.get("/tweets/{user_id}")
def tweets(user_id:int):
    obj=TwittsLookup(user_id)
    result = obj.get()
    return result

@app.get("/wordcloud/{user_id}")
def wordcloud(user_id:int):
    obj=WordCloud(user_id)
    result = obj.get()
    return result

@app.get("/sentiments/{user_id}")
def sentiments(user_id:int):
    obj=Sentiments(user_id)
    result = obj.get()
    return result


# twitts lookup
class TwittsLookup:
    def __init__(self,user_id):
        self.user_id=user_id #155659213
        self.file_name = None

    def getFileName(self):
        return self.file_name
    
    def check_queue(self,name):
        while f'queue_{name}_{self.user_id}' in os.listdir('queue'):
            print(f'{self.user_id} waiting in queue')
            time.sleep(2)
        return True
    def create_queue(self,name):
        f=open(f'queue/queue_{name}_{self.user_id}','w')
        f.close()

        
    def check_exist(self,path):
        for item in os.listdir(path):
            if str(self.user_id) in item:
                self.file_name = item
                p_file=self.file_name.split('_')[1]
                available_date=datetime.datetime.strptime(p_file,'%Y-%m-%d %H:%M:%S.%f')
                print('file date',available_date)
                if (available_date+datetime.timedelta(hours=3)) < datetime.datetime.now(): #data abailable but over 3 hours
                    print('file not exist returning false')
                    return False
                else:
                    print('file available under 3 hrs : returning true')
                    return True
                    

    def destroy_queue(self,name):
        try:
            os.remove(rf'queue/queue_{name}_{self.user_id}') # remove queue.
            print(f'removed queue: queue/queue_{self.user_id}')
        except Exception as e:
            print(f'Error in removing {name} queue')

    def connect_to_endpoint(self,url, tweet_fields):
        response = requests.request(
            "GET", url, auth=bearer_oauth, params=tweet_fields)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(
                "Request returned an error: {} {}".format(
                    response.status_code, response.text
                )
            )
        return response.json()
    def create_url(self,pagination_token=None):
        tweet_fields = "tweet.fields=attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,source,text"
        id = self.user_id
        max_results=100
        start_time = (datetime.datetime.now()- datetime.timedelta(days= 180)).strftime("%Y-%m-%dT%H:%M:%SZ") # last 60 days
        if pagination_token :
            url = "https://api.twitter.com/2/users/{0}/tweets?max_results={1}&start_time={2}&pagination_token={3}".format(id,max_results,start_time,pagination_token)
        else:
            url = "https://api.twitter.com/2/users/{0}/tweets?max_results={1}&start_time={2}".format(id,max_results,start_time)
        return url, tweet_fields
    

    def get_twitts(self):
        self.create_queue(name='tweets')
        url, tweet_fields = self.create_url()
        json_response = self.connect_to_endpoint(url, tweet_fields)
        while 'next_token' in json_response['meta'].keys():
            print('meta----',json_response['meta'])
            url, tweet_fields = self.create_url(pagination_token=json_response['meta']['next_token'])
            next_response = self.connect_to_endpoint(url, tweet_fields)
            json_response['data'].extend(next_response['data']) # extends twitts
            json_response['meta'] = next_response['meta']
        
        # delete old file
        if self.file_name :
            try:
                print('we have old twwts files so trying to remove it')
                os.remove(f'saved/{self.file_name}') # old file exist, remove it.
            except Exception as e:
                print('error wile removing old file',e)
            print(f'after remove: saved/{self.file_name}',os.listdir())

        #write pickle /  save data
        with open(f'saved/{self.user_id}_{datetime.datetime.now()}_.pkl', 'wb') as f:
            pickle.dump(json_response, f)
        
        self.destroy_queue(name="tweets")
        
        return json_response
    

    def get(self):
        if self.check_queue(name='tweets'):   #check if process is running.
            if self.check_exist('saved/'):            #check if tweets file is exists
                if self.file_name :
                    print(f'reading file saved/{self.file_name}')
                    with open(f'saved/{self.file_name}', 'rb') as f:  #read existing data
                        print('file exist-----')
                        data__ = pickle.load(f)
            else:
                
                print('file not exist feetching new tweets---')
                data__ = self.get_twitts()
                

        else:
            data__ = {}
        return data__



#sentiments
class Sentiments(TwittsLookup):
    def __init__(self,user_id):
        self.user_id = user_id
        self.file_name = None
    def get_sentiment(self,tweet):
        text_= re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", tweet.lower())
        sentence = Sentence(text_)
        classifier.predict(sentence)
        #return str(sentence.labels[0]) #.split()[0]
        if(len(sentence.labels) == 0 ): return 'Unclassified'
        return str(sentence.labels[0]).split(" ")[-2]
    
    def get(self):
        super().__init__(self.user_id)
        data__ = super().get()
        tweets_list = data__['data']
        df=pd.DataFrame(tweets_list)
        print('insige sentiment')
        
        #check if tweets exists then read pickle of sentiments
        if super().check_queue(name='tweets'):  # check if tweets is in fetching state
            if super().check_queue(name='sentiments'): # check if sentiments is in processing state
                if super().check_exist('saved/sentiments/') and (super().getFileName() is not None):
                    self.file_name=super().getFileName()
                    print(f'reading file saved/sentiments/{self.file_name}')
                    with open(f'saved/sentiments/{self.file_name}', 'rb') as f:  #read existing data
                        json_dic = pickle.load(f)
                                    
                else:
                    # create sentiment processing queue
                    super().create_queue(name='sentiments')
                    df['sentiment']=df['text'].apply(lambda x: self.get_sentiment(x))
                    df=df[['text','sentiment']]
                    json_dic=df.groupby(by=['sentiment']).count().to_dict()

                    # delete old file
                    if self.file_name :
                        try:
                            print('removing old sentiments file',self.file_name)
                            os.remove(f'saved/sentiments/{self.file_name}') # remove old sentiments
                        except Exception as e:
                            print('error wile removing old sentiment file',e)
                    
                    #save pickle
                    print(f'saving----sentiments saved/sentiments/{self.user_id}_{datetime.datetime.now()}_.pkl')
                    with open(f'saved/sentiments/{self.user_id}_{datetime.datetime.now()}_.pkl', 'wb') as f:
                        pickle.dump(json_dic, f)
                    super().destroy_queue(name = 'sentiments')

        else:
            json_dic = {}

        return json_dic
    



# WordCloud lookup
class WordCloud:
    def __init__(self,user_id):
        self.user_id=user_id #155659213

    def get(self):
        obj = TwittsLookup(self.user_id)
        data__ = obj.get()
        tweets_list = data__['data']
        occurence = {}
        for text in tweets_list:
            text_= re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text['text'].lower())
            clean = nlp(text_)
            words=clean.ents
            for word in words:
                if str(word) not in occurence.keys() and not str(word).isnumeric():    
                    occurence[str(word)] = text_.count(str(word))
        final_dict=dict(sorted(occurence.items(), key=lambda item: item[1],reverse=True))
        return {'data':final_dict}





# following lookup
class FollowLookup:
    def __init__(self,user_id):
        self.user_id=user_id #'155659213'
    def connect_to_endpoint(self,url, user_fields):
        response = requests.request(
            "GET", url, auth=bearer_oauth, params=user_fields)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(
                "Request returned an error: {} {}".format(
                    response.status_code, response.text
                )
            )
        return response.json()
    def create_url(self):
        # Replace with user ID below
        user_field="user.fields=created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified"
        user_id = self.user_id
        url = "https://api.twitter.com/2/users/{}/following".format(user_id)
        return url,user_field
    def get_following(self):
        url,user_field = self.create_url()
        json_response = self.connect_to_endpoint(url,user_field)
        # print(json.dumps(json_response, indent=4, sort_keys=True))
        return json_response
    def get(self):
        data__ = self.get_following()
        return data__



class UserLookup:
    #https://api.twitter.com/2/users/by?usernames=BillGates&
    def __init__(self,username):
        self.username=username #155659213
    def connect_to_endpoint(self,url, tweet_fields):
        response = requests.request(
            "GET", url, auth=bearer_oauth, params=tweet_fields)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(
                "Request returned an error: {} {}".format(
                    response.status_code, response.text
                )
            )
        return response.json()
    
    def create_url(self):

        tweet_fields = "user.fields=created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,url,username,verified,public_metrics"
        username = self.username
        url = "https://api.twitter.com/2/users/by?usernames={}".format(username)
        return url, tweet_fields

    def get(self):
        url, tweet_fields = self.create_url()
        json_response = self.connect_to_endpoint(url, tweet_fields)
        return json_response
