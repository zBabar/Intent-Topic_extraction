import pandas as pd
import spacy
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

# pattern = r'(##.*?##)'

# # Extract all matches and store them in a list
# tokens = test_df['user_msg'].str.extractall(pattern)[0].tolist()

# print(np.unique(tokens))

class Dataset:

    def __init__(self):

        # Read data from the the Excel file
        self.labelled_df = pd.read_excel('./data/Labelled_data.xlsx')
        self.unlabelled_df = pd.read_excel('./data/Unlabelled_data.xlsx')


    def get_labelled_data(self):

        return self.labelled_df[["phrase","label"]]


    def get_unlabelled_data(self):

        return self.unlabelled_df[["user_msg"]]


class Preprocessing:

    def __init__(self):
        
        # Load the Dutch spaCy model
        self.nlp = spacy.load("nl_core_news_sm")

        # Initialize the label encoder
        self.label_encoder = LabelEncoder()

        # Get dutch stop words
        self.dutch_stop_words = stopwords.words('dutch')

        # Vectorize the text data
        self.vectorizer = TfidfVectorizer(stop_words=self.dutch_stop_words)

        self.translate = {
                            '##DATE##': 'DATUM',
                            '##EMAIL##': 'E-MAIL',
                            '##ETHNIC_GROUP##': 'ETNISCHE_GROEP',
                            '##FIRST_NAME##': 'NAAM',
                            '##LAST_NAME##': 'NAAM',
                            '##LOCAT##': 'LOCATIE',
                            '##LOCATION##': 'LOCATIE',
                            '##ON##': 'OP',
                            '##PHONENR##': 'TELEFOONNUMMER',
                            '##STREET_ADDRESS##': 'STRAATADRES'
                        }

    def clean_text(self,text):

        # print(text)
        
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.like_num and not token.like_email and not token.like_url:
                tokens.append(token.lemma_)

        return ' '.join(tokens)

    def get_noun_phrase(self, text):

        doc = self.nlp(text)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]

        return ' '.join(noun_phrases)
    
    def apply_encoder(self, labels):

        return self.label_encoder.fit_transform(labels)

    def get_preprocessd_data(self, data, text_column=None, has_labels = True):

        if(has_labels):

            # Encode the labels
            data['class'] = self.apply_encoder(data['label'])

            # Apply preprocessing to the text column
            data['cleaned_text'] = data[text_column].apply(self.clean_text)
            data['noun_phrase'] = data[text_column].apply(self.get_noun_phrase)

        

        else:
            # Apply preprocessing to the text column
            data = data.dropna()
            data = data.reset_index(drop=True)
            data[text_column] = data[text_column].replace(self.translate, regex=True)

            # Convert the text to lowercase
            data[text_column] = data[text_column].str.lower()
            data['cleaned_text'] = data[text_column].apply(self.clean_text)
            data['noun_phrase'] = data[text_column].apply(self.get_noun_phrase)
        
        return data
    
    def get_DTM(self, data, has_labels = True):

        if(has_labels):

            X = self.vectorizer.fit_transform(data['cleaned_text'])
            y = data['class']

            return X, y

        else:

            X = self.vectorizer.transform(data['cleaned_text'])
        
        return X
    
    def get_tokenized(self, data):
        tokenized_docs = []

        texts = data["cleaned_text"].to_list()
        for text in texts:
            tokenized_doc = []
            doc = self.nlp(text)
            # tokens = []
        
            tokenized_docs.append([token.text for token in doc])

        return tokenized_docs

    def label_decoder(self,labels):

        return self.label_encoder.inverse_transform(labels)
    
    def get_vectorizer(self):

        return self.vectorizer


