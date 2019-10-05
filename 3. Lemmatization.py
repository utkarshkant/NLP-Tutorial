# -*- coding: utf-8 -*-

'''
Part 3
Lemmatization : Same as stemming but the intermediate representation / root form has a meaning.
      Example : 'Intelligence', 'intelligent' & 'intelligently' will be represented by the root word 'intelligent', which has a meaning in the English language.

Q. How Lemmatization is different from Stemming?
A. In lemmatization :
      1. Word reresentations / root form have meaning
      2. Takes more time than stemming
[IMP] 3. Lemmatization is preferred over stemming when meaning of the word is required , ex - QnA applications, Google assistant, Alexa etc
[IMP] 4. Stemming is preferred when the meaning of the word is not require, ex - spam detection, sentiment analysis

Article on differences between Stemming & Lemmatization : https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
'''

# import nltk library
import nltk
# import WordNetLemmatizer module from nltk that'll perform lemmatization
from nltk.stem import WordNetLemmatizer
# import stopwords module
from nltk.corpus import stopwords

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

# convert paragraph into senteces
sentences = nltk.sent_tokenize(paragraph)  # converted para into 31 sentences

# initialise WordNetLemmatizer()
lemmatizer = WordNetLemmatizer()

# perform lemmatization
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])  # create a list of words in each sentence
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)