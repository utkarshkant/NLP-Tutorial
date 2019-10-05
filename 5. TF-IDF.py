# -*- coding: utf-8 -*-

'''
Part 5
Creating Document Matrix by TF-IDF

We will discuss how to create document matrix by TF-IDF i.e. Term Frequency - Inverse Document Frequency.
This methodology helps us overcome the few disadvantages that we face with Bag of Words.

Problems with Bag of Words:
    1. All words have same importance
    2. No semantic information is preserved
    
Solution : TF-IDF
    TF-IDF preserves some semantic information as uncommon words are given more importance than common words
    Example - In the text 'she is beautiful', the word 'beautiful' will be given more importance when compared with 'she' & 'is'

Intuition
TF = Term Frequenccy
IDF = Inverse Document Frequency
TF-IDF = TF * IDF

Term Frequency
TF = Number of occurrences of a word in a document / Total number of words in that document

Example : Calculate the TF for the sentence "to be or not to be"
Solution : 
    Total number of words = 6
    Frequency of "to"     = 2
    Frequency of "be"     = 2
    Frequency of "not"    = 1
    
    TF (to)  = 2/6 = 0.333
    TF (be)  = 2/6 = 0.333
    TF (not) = 1/6 = 0.166
    
Steps in TF:
    1. All steps of bag of words till filtering of most frequent words.
    2. Calculate TF for all sentences and create a matrix (columns are sentences / documents and rows are the individual words)

Inverse Document Frequency
IDF = log(Total Number of documents / Number of documents containing the word)

Example : Calculate IDF for the 3 documents below :
    "to be or not to be"
    "i have to be"
    "you got to be"
Solution :
    Calculating IDF for "to" :
        Total number of documents = 3
        Number of documents containing "to" = 3
        IDF (to) = log(3/3) = 0

    Calculating IDF for "be" :  
        Total number of documents = 3
        Number of documents containing "be" = 3
        IDF (be) = log(3/3) = 0

    Calculating IDF for "have" :  
        Total number of documents = 3
        Number of documents containing "have" = 1
        IDF (have) = log(3/1) = 0.477
        
Steps :
    Similarly, we will create the IDF matrix for all the words in the documents

TF-IDF :
    1. Calculate TF Matrix for documents
    2. Calculate IDF Matrix for documents
    3. Multiply both matrices (TF & IDF) to obtain the TF-IDF Matrix
       TF-IDF (word) = TF(document, word) * IDF(word)
'''

# import nltk library
import nltk

# paragraph from the speech of the great Dr. APJ Abdul Kalam
paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""

    # text cleaning
import re # import regular expression module to convert text into lowercase
# import stopwords module
from nltk.corpus import stopwords
# import WordNetLemmatizer module from nltk that'll perform lemmatization
from nltk.stem import WordNetLemmatizer

wordNet = WordNetLemmatizer()  # initialise WordNetLemmatizer()

# convert paragraph into sentences
sentences = nltk.sent_tokenize(paragraph)
corpus = []  # to store sentences here after cleaning

# loop to perform cleaning (stopwords removal + lemmatization) on sentences
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i]) # substitute every character except a to z and A to Z with space (' ')
    review = review.lower()  # convert everything into lowercase
    review = review.split()  # create a list of words by splitting the sentence on space (' ')
    review = [wordNet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]  # remove stopwords then perform lemmatization
    review = ' '.join(review)  # re-join the processed words into a sentence
    corpus.append(review)  # append the processed sentences in the list - corpus

# creating TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1500)  # initialise TfidfVectorizer
X = tfidf.fit_transform(corpus).toarray()  # apply tfidf to corpus and transform into array to create TF-IDF Document Matrix

