import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option('display.width', 200)

nltk.download("stopwords")
stop_words = stopwords.words("english")
nltk.download('punkt')

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
pos_word_list = []
neu_word_list = []
neg_word_list = []


def data_collection():
    print('* * * * * Data Collection * * * * *')
    df_all = pd.DataFrame()

    # load data set
    df = pd.read_csv('uploads/Chat with Uni fellows.csv')
    df = df.assign(Chat_With='Univ Fellows')
    df = df.assign(Sentiment_Score='')
    # print(new_df.head())
    # get number of records in data set and columns
    # print(df.shape)
    df_all = pd.concat([df_all, df], ignore_index=True)

    # load data set
    df = pd.read_csv('uploads/Chat with Close friends.csv')
    df = df.assign(Chat_With='Close Friends')
    df = df.assign(Sentiment_Score='')
    # get number of records in data set and columns
    # print(df.shape)
    df_all = pd.concat([df_all, df], ignore_index=True)

    print(df_all.shape)
    print(df_all.head())
    return df_all


def pre_processing(df):
    print('* * * * * Pre-Processing * * * * *')

    # data cleaning
    # get data types of columns
    # also check for null values
    print(df.info())

    # As we are going to do a NLP project so,
    # we need only ratings and reviews columns.
    # We will drop rest of the columns!
    df.drop(['Day', 'Date', 'Time', 'User Name', 'Media'],
            axis=1, inplace=True)
    df.columns = ['Message', 'Chat_With', 'Sentiment_Score']
    print(df.head())

    # check null values
    print(df.isnull().sum())
    print(df.shape)

    df.dropna(subset=['Message'], inplace=True)
    print(df.shape)

    return df


def text_pre_processing(df):
    print('* * * * * Text Pre-Processing * * * * *')
    df['Message'] = df['Message'].apply(remove_punctuation)
    # print(df.head())
    df['Message'] = df['Message'].apply(remove_numbers)
    # print(df.head())
    df['Message'] = df['Message'].apply(remove_stopwords)
    return df


def get_sentiment_score(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment


def remove_punctuation(df):
    text = re.sub("n't", 'not', df)
    text = re.sub('[^\w\s]', '', df)
    return text


def token(df):
    tokenized_text = word_tokenize(df)
    return tokenized_text


def remove_numbers(df):
    result = ''.join([i for i in df if not i.isdigit()])
    return result


def remove_stopwords(df):
    tokens = token(df)
    result = ' '.join([w for w in tokens if w not in stop_words])
    return result


def populate_positive_negative_neutral_words():
    for tokens in df['Message'].apply(token):
        for word in tokens:
            # print(word)
            if (analyzer.polarity_scores(word)['compound']) >= 0.5:
                pos_word_list.append(word)
            elif (analyzer.polarity_scores(word)['compound']) <= -0.5:
                neg_word_list.append(word)
            else:
                neu_word_list.append(word)

    print('Positive :', pos_word_list)
    print('Neutral :', neu_word_list)
    print('Negative :', neg_word_list)


if __name__ == '__main__':
    df = data_collection()
    df = pre_processing(df)
    df = text_pre_processing(df)

    # apply get sentiment score
    df['Sentiment_Score'] = df['Message'].apply(get_sentiment_score)
    print(df.head())

    populate_positive_negative_neutral_words()
