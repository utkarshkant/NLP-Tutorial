# -*- coding: utf-8 -*-

'''
Part 4
Creating Document Matrix by Bag of Words Methodology: 
    Step 1 - Convert all words in sentences into lowercase.
             This is because there can be multiple words in the paragraph that are same but will be considered different due to the difference in the upper & lower case letters.
             Therefore, we convert all words into lowercase for equal treatment.
    Step 2 - Tokenizations
             Convert paragraph into sentences and words.
    Step 3 - Histogram
             Create a matrix that records the words in our corpus and maintains their count or the frequency of the word elements
    Step 4 - Sort the Histogram in a descending order of the word frequency
    Step 5 - Filter words
             Filter out / remove the words that do not appear very frequently as their contribution to our analysis will be very negligible
             Retain the words that have a higher frequency, the top-most frequent words
        ex - select the top 10 frequently occuring words from a set of 14 words; or
             select top 2500 to 3000 frequently occuring words from a set of 6000 words
    Step 6 - Create the Matrix (Document Matrix) or the Bag of Words
             The individual words make up the column of the matrix (features)
             and the sentences make up the rows of the matrix.
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
# import module to stem our text
from nltk.stem import PorterStemmer
# import WordNetLemmatizer module from nltk that'll perform lemmatization
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()  # initialise PorterStemmer()
wordNet = WordNetLemmatizer()  # initialise WordNetLemmatizer()

# convert paragraph into sentences
sentences = nltk.sent_tokenize(paragraph)
corpus = []  # store sentences here after cleaning

# loop to perform cleaning (stopwords removal + stemming / lemmatization) on sentences
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i]) # substitute every character except a to z and A to Z with space (' ')
    review = review.lower()  # convert everything into lowercase
    review = review.split()  # create a list of words by splitting the sentence on space (' ')
    # review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  # remove stopwords then perform stemming
    review = [wordNet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]  # remove stopwords then perform lemmatization
    review = ' '.join(review)  # re-join the processed words into a sentence
    corpus.append(review)  # append the processed sentences in the list - corpus
    
'''
Comparing the sentences in the lists `corpus` & `sentences` we realise the differences in each sentence.
Now let us perform the same operation with lemmatization instead of stemming. Code above.
'''

# creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)  # initialise CountVectorizer
X = cv.fit_transform(corpus).toarray()  # apply cv to corpus and transform into array to create bag of words
'''
Note that X has dimensions of 31 x 114, which means it contains 31 sentences and 114 unique words
'''