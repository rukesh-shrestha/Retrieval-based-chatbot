# importing libraries and dataset.
from flask import Flask,render_template,request
from chatbot.configuration import DATAFILE
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random
from keras.models import load_model
app = Flask(__name__)


model = load_model("model.h5") #loading save model.
intents = json.loads(open(DATAFILE,encoding='utf-8').read())  #reading the json data.
words = pickle.load(open("words.pkl","rb")) #loading save pickle words.
classes = pickle.load(open("classes.pkl","rb")) #loading save classes.

def cleansentences(sentence):
    """
    :param sentence: takes the query ask by client
    :return: list of lemmatizer word
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence,words,show_details=False):
    """

    :param sentence: takes the query ask by client
    :param words: takes the words we have pickle, it is the list of tokenized, lemmantized word
    :param show_details: it is the boolean expression
    :return: array of bag of words.
    """
    sentence_words = cleansentences(sentence)
    bag = [0] * len(words)
    # print(words)
    # print(bag)
    for sentences in sentence_words:
        for key,values in enumerate(words):
            # print("key",key,": value",values)
            if values == sentences:
                bag[key] = 1
                if show_details:
                    print("Found in bag : %S" % values)
    # print(bag)
    return (np.array(bag))



def predictclasses(sentence):
    """

    :param sentence: takes the query ask by client
    :return: list with the word and its probability.
    """
    result = model.predict(np.array([bow(sentence,words)]))[0]
    results = [[key,value] for key,value in enumerate(result)]
    results.sort(key=lambda x: x[1], reverse=True)
    returnlist = list()
    for eachresult in results:
        eachr = [classes[eachresult[0]]],[eachresult[1]]
        returnlist.extend(eachr)
    return returnlist




@app.route('/')
def home():
    """

    :return: home page
    """
    return render_template("home.html")

@app.route("/get")
def get_box_response():
    """
    This function is used to get message from the frontend and send back responses to it.
    :return: string of responses.
    """
    inp = request.args.get('msg')

    results = predictclasses(sentence=inp)
    resultsindex = np.array(results)
    tag = resultsindex[0]
    listofintents = intents['intents']
    for i in listofintents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return str(result)





if __name__ == '__main__':
    """
    This is the part where the program is executed. 
    
    """
    app.run(debug=True)
