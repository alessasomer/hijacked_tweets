import tweepy
import config
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer
import pickle

class Test:
    def __init__(self):
        self._nested = []
        self._tweet_examples = []

    def create_tweetcsv_test(self):
       trainingList =[]
       with open('test.txt') as file_object:
           for jsonObj in file_object:
               trainingDict = json.loads(jsonObj)
               trainingList.append(trainingDict)

       api_key = 'YEpNeLOhlYEL5OId1bCKenHxD'
       api_key_secret = 'AdWfTqltmr7qKSOkOjujkipCGs08UuMjegYRLqpHkLT6Fv362L'
       BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAOJnAEAAAAAyVKYaq6rfDF7NeHLThQ86hVJibk%3DatFtfDHbOAYzA3IcsLvfECssqqrk8uS4d8967dJFgD7Cs4dlCU'


       access_token = "1651328402245185538-7yv4hOJshKAnLroHWEmq0T47TRdFML"
       access_token_secret = "aWpwR6zzZsdqU6GixgTC8nuLDkr73jcGL83vKNDh259JM"


       #authentication
       auth = tweepy.OAuthHandler(api_key, api_key_secret)
       auth.set_access_token(access_token, access_token_secret)


       #call Api
       api = tweepy.API(auth)


       #create dataframe
       columns = ['Label', 'Tweet']
       data = []
       for t in trainingList:
           try:
               status = api.get_status(t["id"])
               #get text
               text = status.text
               data.append([t["label"], text])
           except:
               #print("Tweet with ID" , t["id"] , "does not exist")
               continue
       df = pd.DataFrame(data, columns=columns)
       test_label_examples = df['Label'].tolist()
       test_tweet_examples = df['Tweet'].tolist()
       print(test_tweet_examples)
       #df.to_csv('tweets.csv')
       #NOTE: here tweet_examples has not been tokenized yet, and is a list of tweet texts NOT a list of words
       return test_tweet_examples, test_label_examples
    
def load_preprocessed_data():
    with open('mypickle.pickle', 'rb') as f:
       data = []
       while True:
            try:
               data.append(pickle.load(f))
            except EOFError:
                break
    print(data)
    return data

def get_data():
    items = list(load_preprocessed_data())
    X_train = items[0]
    Y_train = items[1]
    X_dev = items[2]
    Y_dev = items[3]
    return X_train, Y_train, X_dev, Y_dev
        

def main():
    test = Test()
    #tup = test.create_tweetcsv_test()
    #load_preprocessed_data()
    #get_data()
    file = open('mypickle.pickle', 'rb')
    # dump information to that file
    data = pickle.load(file)
    # close the file
    file.close()
    print(data[2])
    #f = open('mypickle.pickle', 'rb')
    #favorite_color = None
    #while True:
            #try:
               #favorite_color = pickle.load(f) 
            #except EOFError:
                #break
    #print(favorite_color)
    

if __name__ == '__main__':
    main()
