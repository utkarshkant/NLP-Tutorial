# -*- coding: utf-8 -*-

'''
Part 2
Stemming : Stemming is a process of reducing infected or derived words to their word stem, base or root form
For example : 'Intelligent', 'Intelligence' & 'Intelligently' all three words stem from a root word - 'Intelligen'.
              Therefore, 'Intelligen' is the root word for all the above 3 words
Example 2 : Words 'going', 'goes' & 'gone' all stem from the root word - 'go'
'''

# import nltk library for text pre-processing
import nltk
# import module to stem our text
from nltk.stem import PorterStemmer
# import module to remove stopwords
from nltk.corpus import stopwords
'''
Stopwords : These are the words like 'of', 'the', 'they' that are always present in a text paragraph
            These stopwords do not contribute in the analysis of the paragraph, which is why, they must be removed
Note - `stopwords.words('english')` gives you the list of all stopwords in the English language
'''

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

# initialise PorterStemmer()
stemmer = PorterStemmer()

# perform stemming over the sentences
# this for loop removes stopwords from the sentence and performs stemming on the inlusive words
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])  # create a list of words in each sentence
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]  # list comprehension to check if the word doesn't belong to the stopwords and then perform stemming procedure on that word
    sentences[i] = ' '.join(words)  # rejoin the words to re-create the sentence
    
'''
The problem with Stemming : Stemming produces intermediate representation of the word that may not have any meaning.
                            i.e. the root word produced may not have any meaning
                  Example : 'intelligen', 'fina', 'histori' and so on ...
                 Solution : Lemmatization
'''
