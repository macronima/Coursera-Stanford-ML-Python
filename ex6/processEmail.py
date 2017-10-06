import re
import getVocabList as gvl
from stemming.porter2 import stem
from getVocabList import getVocabList
# This porter stemmer seems to more accurately duplicate the
# porter stemmer used in the OCTAVE assignment code
# (note: I had to do a pip install nltk)
# I'll note that both stemmers have very similar results
import nltk, nltk.stem.porter
import numpy as np


def preProcess(email):
    """
    Function to do some pre processing (simplification of e-mails).
    Comments throughout implementation describe what it does.
    Input = raw e-mail
    Output = processed (simplified) email
    """
    # Make the entire e-mail lower case
    email = email.lower()

    # Strip html tags (strings that look like <blah> where 'blah' does not
    # contain '<' or '>')... replace with a space
    email = re.sub('<[^<>]+>', ' ', email);

    # Any numbers get replaced with the string 'number'
    email = re.sub('[0-9]+', 'number', email)

    # Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)

    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email);

    # The '$' sign gets replaced with 'dollar'
    email = re.sub('[$]+', 'dollar', email);

    return email


def email2TokenList(rawEmail):
    """
    Function that takes in preprocessed (simplified) email, tokenizes it,
    stems each word, and returns an (ordered) list of tokens in the e-mail
    """

    # I'll use the NLTK stemmer because it more accurately duplicates the
    # performance of the OCTAVE implementation in the assignment
    stemmer = nltk.stem.porter.PorterStemmer()

    email = preProcess(rawEmail)

    # Split the e-mail into individual words (tokens) (split by the delimiter ' ')
    # but also split by delimiters '@', '$', '/', etc etc
    # Splitting by many delimiters is easiest with re.split()
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)

    # Loop over each word (token) and use a stemmer to shorten it,
    # then check if the word is in the vocab_list... if it is,
    # store what index in the vocab_list the word is
    tokenlist = []
    for token in tokens:

        # Remove any non alphanumeric characters
        token = re.sub('[^a-zA-Z0-9]', '', token);

        # Use the Porter stemmer to stem the word
        stemmed = stemmer.stem(token)

        # Throw out empty tokens
        if not len(token): continue

        # Store a list of all unique stemmed words
        tokenlist.append(str(stemmed))

    return tokenlist

def email2VocabIndices( rawEmail, vocabDict ):

    tokenlist = email2TokenList( rawEmail )
    indecesList = []
    #l = length(vocabDict
    for token in tokenlist:
        if token in vocabDict:
            indecesList.append(vocabDict.index(token))
 #   indecesList = [ vocabDict[token] for token in tokenlist if token in vocabDict ]
    qq=0
    return indecesList


def email2FeatureVector( rawEmail, vocabDict ):

    n = len(vocabDict)
    result = np.zeros((n,1))
    vocab_indices = email2VocabIndices( rawEmail, vocabDict )
    for idx in vocab_indices:
        result[idx] = 1
    return result


def processEmail(emailContent):
    vocabList = getVocabList()
    indecesList = email2VocabIndices(emailContent,vocabList)
    featureList = email2FeatureVector(emailContent,vocabList)
    return indecesList