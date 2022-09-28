import pandas as pd
import stanza
import re
import threading
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import FreqDist
import stanfordnlp
import random
from nltk.tag import StanfordNERTagger
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_sql(data)
df['text'] = df['text'].str.strip()


# Define function to find certain attributes

def find_attribute(attributes, text):
    '''
    :return: Each attribute in a line of commentary
    '''
    return next((x for x in attributes if x in text), "")


# Define the possible penalty foot, location, and outcomes
feet = ['left foot', 'right foot']
locations = ['bottom left', 'bottom right', 'top right', 'top left', 'center', 'centre', 'post', 'crossbar', 'miss',
             'lower left', 'lower right', 'upper right', 'upper left', 'middle']
outcomes = ['convert', 'save', 'miss', 'post', 'bar', 'goal', 'scores']

var_dict = {'foot': feet,
            'location': locations,
            'event_outcome': outcomes
            }
# Find the words or phrases in each line of commentary that match the feet, locations, and outcomes, and  store the result
for key, value in var_dict.items():
    df['{}'.format(key)] = df.apply(lambda x: find_attribute(value, x['text']), axis=1)

df.reset_index(inplace=True)

# Replace common grammatical error in data to make full sentence

target_phrases = [',  right footed shot ', ',  left footed shot ']
true_phrases = [', his/her right footed shot is ', ', his/her left footed shot is ']

target = df['text'].str.contains(target_phrases[0]), 'text'
df.loc[target] = df.loc[target].str.replace(target_phrases[0], true_phrases[0])
df.loc[target] = df.loc[target].str.replace(target_phrases[1], true_phrases[1])

# Download Spacy and Tokenize
nlp = stanza.Pipeline(lang='en', processors={
    'tokenize': 'spacy'})  # spaCy tokenizer is currently only allowed in English pipeline.

def get_taker_and_team(text):
    '''
    Function parses text of each commentary and returns the taker and team
    :param text:
    :return:
    team = team doing the action
    taker = person doing the action
    :raises:
    IndexError: some commentaries do not contain the team doing the action. These are returned as NaN
    '''
    doc = nlp(text)
    try:
        taker_fw = [word.id for word in doc.sentences[-1].words if (word.deprel == 'nsubj')][0]
        taker = [word.text for word in doc.sentences[-1].words if
                 (word.deprel == 'nsubj' or word.head == taker_fw) and word.deprel != 'appos']
        taker = (' '.join(map(str, taker)))
        team_fw = [word.id for word in doc.sentences[-1].words if (word.deprel == 'appos')][0]
        team = [word.text for word in doc.sentences[-1].words if
                (word.deprel == 'appos' or word.head == team_fw) and word.deprel != 'punct']
        team = (' '.join(map(str, team)))

    except IndexError:
        taker = np.NAN
        team = np.NAN

    return taker, team

# Apply function to text
df[['taker', 'team']] = pd.DataFrame(df['text'].progress_apply(get_taker_and_team).tolist(),
                                              index=df.index)

# Simplify the types of outcomes
true_outcome = {'convert': 'goal',
                'goal': 'goal',
                'miss': 'miss',
                'post': 'miss',
                'bar': 'miss',
                'save': 'save'}

true_loc = {'bottom left': 'bottom left',
            'bottom right': 'bottom right',
            'top right': 'top right',
            'top left': 'top left',
            'center': 'center',
            'centre': 'center',
            'post': 'miss',
            'crossbar': 'miss',
            'miss': 'miss',
            'lower left': 'bottom left',
            'lower right': 'bottom right',
            'upper right': 'top right',
            'upper left': 'top left',
            'middle': 'center'
            }
# Change locations to common locations
df['location'] = df['location'].map(true_loc)
df['event_outcome'] = df['event_outcome'].map(true_outcome)



