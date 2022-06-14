import csv
import json
import re
import sqlite3
from hashlib import md5
from gensim.parsing.preprocessing import remove_stopwords
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('wordnet')
spacy.load("en_core_web_sm")

VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'Y'}



def memory_caching(foo):
    cache_data = {}

    def wrapped(arg):
        if arg not in cache_data:
            cache_data[arg] = foo(arg)
        return cache_data[arg]

    return wrapped


@memory_caching
def to_binary(syllable):
    if any(vowel in syllable for vowel in VOWELS):
        # If a syllable is stressed - 1 in pattern
        return '1' if syllable[-1] == '1' else '0'

    return ''


def db_caching(foo):
    idx = {'idx': 0}
    conn = sqlite3.connect('data/example.db')
    c = conn.cursor()
    fetched = c.execute(f'SELECT hashed, value from tfidfpreprocess').fetchall()
    cached = {i[0]: json.loads(i[1]) for i in fetched}

    def wrapped(arg):
        hashed = md5(arg.encode()).hexdigest()
        value = cached.get(hashed)
        if not value:
            value = foo(arg)
            c.execute(f'INSERT INTO tfidfpreprocess values (?, ?)', (hashed, json.dumps(value)))
            conn.commit()

        idx['idx'] += 1
        return value

    return wrapped


# Function to preprocess text for TF-IDF or Counter vectorizers
@db_caching
def preprocess(text):
    # Tokenise words while ignoring punctuation
    tokeniser = nltk.RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(text)

    # Lowercase and lemmatise
    lemmatiser = nltk.WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]

    # Remove stopwords
    keywords = [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
    return keywords


# Transformer of rhythm 1
class RhythmTransformer1(TransformerMixin):
    def __init__(self, value=None):
        TransformerMixin.__init__(self)
        self.binary_words_map = {}
        self.value = value

    def fit(self, *_):
        return self

    # Changed this to only return new column after some operation on X
    def transform(self, X):
        data = list(X.values)
        result = []
        for idx, d in enumerate(data):
            result.append(self.get_lyrics_coordinates(d))

        return result

    binary_words_map = {}

    def prepare_binary_words(self):
        cmu_data = list(csv.reader(open('data/CMUdict.csv', 'r')))
        for data in cmu_data[1:]:
            self.binary_words_map[data[0]] = ''.join(to_binary(syllable) for syllable in data[1:] if syllable)

    def get_lyrics_coordinates(self, lyrics):
        if not self.binary_words_map:
            self.prepare_binary_words()

        # Split lines
        split_lines = re.split('\n', lyrics)
        lyrics = [nltk.RegexpTokenizer(r"[\'\w\-]+").tokenize(i) for i in split_lines]

        # Get pattern of strings:
        song_pattern = []
        result = ''
        for line in lyrics:
            for word in line:
                tmp = self.binary_words_map.get(word.upper(), '_')
                result += tmp
            song_pattern.append(result)
            result = ''

        # Count coordinates
        song_1, song_0, song_missing = 0, 0, 0
        for line in song_pattern:
            song_1 += line.count("1")
            song_0 += line.count("0")
            song_missing += line.count("_")
        count_coordinates = [song_1, song_0, song_missing]

        # Count percentage
        total = song_1 + song_0 + song_missing
        coordinates = [(i / total) for i in count_coordinates]
        return coordinates


# Transformer of rhythm 2
class RhythmTransformer2(TransformerMixin):
    def __init__(self, value=None):
        TransformerMixin.__init__(self)
        self.binary_words_map = {}
        self.value = value

    def fit(self, *_):
        return self

    # Changed this to only return new column after some operation on X
    def transform(self, X):
        data = list(X.values)
        result = []
        for idx, d in enumerate(data):
            result.append(self.get_lyrics_coordinates(d))

        return result

    binary_words_map = {}

    def prepare_binary_words(self):
        cmu_data = list(csv.reader(open('data/CMUdict.csv', 'r')))
        for data in cmu_data[1:]:
            self.binary_words_map[data[0]] = ''.join(to_binary(syllable) for syllable in data[1:] if syllable)

    def get_lyrics_coordinates(self, lyrics):
        if not self.binary_words_map:
            self.prepare_binary_words()

        # Split lines
        split_lines = re.split('\n', lyrics)
        lyrics = [nltk.RegexpTokenizer(r"[\'\w\-]+").tokenize(i) for i in split_lines]

        line_syl_len = []
        word_syl_len = []

        line_len = []
        word_len = []
        line_counter = 0

        # Get pattern of strings:
        song_pattern = []
        result = ''
        for line in lyrics:
            for word in line:
                tmp = self.binary_words_map.get(word.upper(), '_')
                result += tmp
                word_syl_len.append(len(tmp))
                word_len.append(len(word))
                line_counter += len(word)
            song_pattern.append(result)
            line_syl_len.append(len(result))
            line_len.append(line_counter)
            line_counter = 0
            result = ''

        # Count coordinates
        song_1, song_0, song_missing = 0, 0, 0
        for line in song_pattern:
            song_1 += line.count("1")
            song_0 += line.count("0")
            song_missing += line.count("_")
        count_coordinates = [song_1, song_0, song_missing]

        # Count percentage
        total = song_1 + song_0 + song_missing
        coordinates = [(i / total) for i in count_coordinates]
        # Percentage of foreign words
        coordinates.append(count_coordinates[2] / len(word_syl_len))
        # Number of lines
        coordinates.append(len(line_syl_len))
        # Average number of syllables in line
        coordinates.append(sum(line_syl_len) / len(line_syl_len))
        # Number of words
        coordinates.append(len(word_syl_len))
        # Average number of syllables in word
        coordinates.append(sum(word_syl_len) / len(word_syl_len))
        # Average number of letters per line
        coordinates.append(sum(line_len) / len(line_len))
        # Average number of letters per word
        coordinates.append(sum(word_len) / len(word_len))
        return coordinates


# Transformer of parts of speech 1
class PartsOfSpeechTransformer1(TransformerMixin):
    def __init__(self, value=None):
        TransformerMixin.__init__(self)
        self.value = value
        self.nlp = spacy.load('en_core_web_sm')

    def fit(self, *_):
        return self

    def transform(self, X):
        data = list(X.values)
        result = []
        cached = json.load(open('data/parts_of_speech.json', 'r'))
        c = 0
        for idx, d in enumerate(data):
            hashed = md5(d.encode()).hexdigest()
            if hashed not in cached:
                c += 1
                print(c)
                cached[hashed] = self.parts_of_speech_percentages(d)

            result.append(cached[hashed][:-2])

        json.dump(cached, open('data/parts_of_speech.json', 'w'))
        return result

    def parts_of_speech_percentages(self, lyrics):
        """Returns an array with percentages of parts of speech in a song"""
        lyrics = remove_stopwords(lyrics)
        nouns, adjectives, verbs, pronouns, adverbs, foreign_words = 0, 0, 0, 0, 0, 0
        total = 0
        doc = self.nlp(lyrics)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                nouns += 1
            elif token.pos_ == 'ADJ':
                adjectives += 1
            elif token.pos_ == 'VERB':
                verbs += 1
            elif token.pos_ in ['PRON', 'DET']:
                pronouns += 1
            elif token.pos_ in ['ADV', 'ADP']:
                adverbs += 1
            elif token.pos_ == 'X':
                foreign_words += 1
            total += 1

        result = [nouns / total, adjectives / total, verbs / total, pronouns / total, adverbs / total,
                  foreign_words / total]
        return result


# Transformer of parts of speech 2
class PartsOfSpeechTransformer2(TransformerMixin):
    def __init__(self, value=None):
        TransformerMixin.__init__(self)
        self.value = value
        self.nlp = spacy.load('en_core_web_sm')

    def fit(self, *_):
        return self

    def transform(self, X):
        data = list(X.values)
        result = []
        cached = json.load(open('data/parts_of_speech.json', 'r'))
        c = 0
        for idx, d in enumerate(data):
            hashed = md5(d.encode()).hexdigest()
            if hashed not in cached:
                c += 1
                print(c)
                cached[hashed] = self.parts_of_speech_percentages(d)

            result.append(cached[hashed][:-2])

        json.dump(cached, open('data/parts_of_speech.json', 'w'))
        return result

    def parts_of_speech_percentages(self, lyrics):
        """Returns an array with percentages of parts of speech in a song"""
        lyrics = remove_stopwords(lyrics)
        nouns, adjectives, verbs, pronouns, adverbs, foreign_words = 0, 0, 0, 0, 0, 0
        total = 0
        doc = self.nlp(lyrics)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                nouns += 1
            elif token.pos_ == 'ADJ':
                adjectives += 1
            elif token.pos_ == 'VERB':
                verbs += 1
            elif token.pos_ in ['PRON', 'DET']:
                pronouns += 1
            elif token.pos_ in ['ADV', 'ADP']:
                adverbs += 1
            elif token.pos_ == 'X':
                foreign_words += 1
            total += 1

        result = [nouns / total, adjectives / total, verbs / total, pronouns / total, adverbs / total,
                  foreign_words / total]
        return result


# Instantiate data transformers
rhythm_transformer_1 = RhythmTransformer1()
rhythm_transformer_2 = RhythmTransformer2()
parts_of_speech_transformer_1 = PartsOfSpeechTransformer1()
parts_of_speech_transformer_2 = PartsOfSpeechTransformer2()
tfidf_transformer = TfidfVectorizer(analyzer=preprocess)
counter_transformer = CountVectorizer(analyzer=preprocess)
