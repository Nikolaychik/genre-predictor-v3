import time

import pandas as pd
from sklearn.pipeline import FeatureUnion

from classifiers import ln_classifier, nb_classifier, classifier_controller
from transformers import rhythm_transformer_1, rhythm_transformer_2, parts_of_speech_transformer_1, \
    parts_of_speech_transformer_2, tfidf_transformer, counter_transformer


class Facade:
    def main(self):
        df = self.prepare_dataframe()
        pipeline = self.input_pipeline()
        classifier = self.input_classifier()
        self.classify(df, pipeline, classifier)

    def prepare_dataframe(self):
        df = pd.read_csv('data/original_cleaned_lyrics.csv')
        df = df[~df['genre'].isin(['Other', 'Indie', 'R&B', 'Folk', None])]
        df = df[df['lyrics'].notnull()]
        df = df[df['lyrics'].str.strip().str.lower() != 'instrumental']
        df = df[~df['lyrics'].str.contains(r'[^\x00-\x7F]+')]
        df = df[df['lyrics'].str.strip() != '']
        df = df[df['genre'].str.lower() != 'not available']
        df = df[df['lyrics'].str.contains('\w')]
        return df

    def input_pipeline(self):
        features = []
        choice = input('Select your data transformers:\n'
                       '  R1 - for Rhythm1; or R2 - for Rhythm2\n'
                       '  PS1 - for PartsOfSpeech1; or PS2 - for PartsOfSpeech2\n'
                       '  TF - for TFIDFVectorizer; or CNT - for CounterVectorizer\n'
                       'You can input multiple codes separated by space\n')
        for code in choice.split(' '):
            if code == 'R1':
                features.append(('rhythm1', rhythm_transformer_1))
            elif code == 'R2':
                features.append(('rhythm2', rhythm_transformer_2))
            elif code == 'PS1':
                features.append(('parts_of_speech1', parts_of_speech_transformer_1))
            elif code == 'PS2':
                features.append(('parts_of_speech2', parts_of_speech_transformer_2))
            elif code == 'TF':
                features.append(('tfidf', tfidf_transformer))
            elif code == 'CNT':
                features.append(('counter', counter_transformer))
            else:
                print('Wrong input, please try again')
                return self.input_pipeline()

        return FeatureUnion(features)

    def input_classifier(self):
        choice = input('Select a classifier:\n  LN - for Linear Regression\n  NB - for Naive Bayesian\n')
        if choice == 'LN':
            return ln_classifier
        elif choice == 'NB':
            return nb_classifier
        else:
            print('Wrong input, please try again')
            return self.input_classifier()

    def classify(self, df, pipeline, classifier):
        print('Starting the classification process, please wait...\n')
        ts = time.time()
        score = classifier_controller.classify(df, pipeline, classifier)
        print(f'Done! Classification took {int(time.time() - ts)} seconds, the accuracy is:\n{score}')


if __name__ == '__main__':
    Facade().main()
