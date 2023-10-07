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
stop_words.append('Are')
stop_words.append('u')
stop_words.append('I')
nltk.download('punkt')

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
positive_words = []
neutral_words = []
negative_words = []
chat_with_univ_fellows_most_used_word = {}
chat_with_close_friends_most_used_word = {}


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
    df.drop(['Day', 'Time', 'Media'],
            axis=1, inplace=True)
    df.columns = ['Date', 'User Name', 'Message', 'Chat_With', 'Sentiment_Score']
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
    print('* * * * * Positive, Negative & Neutral Words * * * * *')
    for tokens in df['Message'].apply(token):
        for word in tokens:
            # print(word)
            if (analyzer.polarity_scores(word)['compound']) >= 0.5:
                positive_words.append(word)
            elif (analyzer.polarity_scores(word)['compound']) <= -0.5:
                negative_words.append(word)
            else:
                neutral_words.append(word)

    print('Positive :', positive_words)
    print('Neutral :', neutral_words)
    print('Negative :', negative_words)


def populate_most_used_words():
    print('* * * * * Most used Words in each Group * * * * *')
    df_univ_fellows = df[df['Chat_With'] == 'Univ Fellows']
    for tokens in df_univ_fellows['Message'].apply(token):
        for word in tokens:
            item = chat_with_univ_fellows_most_used_word.get(word)
            # print('item: ', item)
            if item is None:
                chat_with_univ_fellows_most_used_word[word] = 1
            else:
                item = item + 1
                chat_with_univ_fellows_most_used_word[word] = item

    df_close_friends = df[df['Chat_With'] == 'Close Friends']
    for tokens in df_close_friends['Message'].apply(token):
        for word in tokens:
            item = chat_with_close_friends_most_used_word.get(item)
            if item is None:
                chat_with_close_friends_most_used_word[word] = 1
            else:
                item = item + 1
                chat_with_close_friends_most_used_word[word] = item

    # print('chat_with_univ_fellows_most_used_word :', chat_with_univ_fellows_most_used_word)
    # print('chat_with_close_friends_most_used_word :', chat_with_close_friends_most_used_word)

    univ_fellows_sorted_keys = sorted(chat_with_univ_fellows_most_used_word,
                                      key=chat_with_univ_fellows_most_used_word.get,
                                      reverse=True)
    print('Most used words in Chat with University Fellows Group')
    i = 0
    for r in univ_fellows_sorted_keys:
        print(r, chat_with_univ_fellows_most_used_word[r])
        i = i + 1
        if i == 10:
            break

    close_friends_sorted_keys = sorted(chat_with_close_friends_most_used_word,
                                       key=chat_with_close_friends_most_used_word.get,
                                       reverse=True)
    print('Most used words in Chat with Close Friends Group')
    i = 0
    for r in close_friends_sorted_keys:
        print(r, chat_with_close_friends_most_used_word[r])
        i = i + 1
        if i == 10:
            break


def populate_most_active_user():
    print('* * * * * Most Active user in each Group * * * * *')
    users_in_univ_fellows = {}
    df_univ_fellows = df[df['Chat_With'] == 'Univ Fellows']
    for user in df_univ_fellows['User Name']:
        item = users_in_univ_fellows.get(user)
        # print('item: ', item)
        if item is None:
            users_in_univ_fellows[user] = 1
        else:
            item = item + 1
            users_in_univ_fellows[user] = item

    users_in_close_freinds = {}
    df_close_friends = df[df['Chat_With'] == 'Close Friends']
    for user in df_close_friends['User Name']:
        item = users_in_close_freinds.get(user)
        if item is None:
            users_in_close_freinds[user] = 1
        else:
            item = item + 1
            users_in_close_freinds[user] = item

    # print(users_in_univ_fellows)
    # print(users_in_close_freinds)
    univ_fellows_sorted_keys = sorted(users_in_univ_fellows,
                                      key=users_in_univ_fellows.get,
                                      reverse=True)
    i = 0
    for r in univ_fellows_sorted_keys:
        print('Most Active User in Chat with University Fellows Group: ', r)
        i = i + 1
        if i == 1:
            break

    close_friends_sorted_keys = sorted(users_in_close_freinds,
                                       key=users_in_close_freinds.get,
                                       reverse=True)
    i = 0
    for r in close_friends_sorted_keys:
        print('Most Active User in Chat with Close Friends Group: ', r)
        i = i + 1
        if i == 1:
            break


def per_day_messages_by_each_memeber_in_each_group():
    print('* * * * * Per Day Messages by each Memeber in each Group * * * * *')
    df_univ_fellows = df[df['Chat_With'] == 'Univ Fellows']
    unique_days_in_univ_fellows = df_univ_fellows['Date'].unique()
    unique_users_in_univ_fellows = df_univ_fellows['User Name'].unique()
    # print(unique_days_in_univ_fellows)
    # print(unique_users_in_univ_fellows)
    print('University Fellows Group')
    for date in unique_days_in_univ_fellows:
        df_filter_by_date = df_univ_fellows[df_univ_fellows['Date'] == date]
        print('Date: ', date)
        for user in unique_users_in_univ_fellows:
            df_filter_by_date_by_user = df_filter_by_date[df_filter_by_date['User Name'] == user]
            print('\tUser:', user, ',\tNo. of Messages: ', len(df_filter_by_date_by_user))

    df_close_friends = df[df['Chat_With'] == 'Close Friends']
    unique_days_in_close_friends = df_close_friends['Date'].unique()
    unique_users_in_close_friends = df_close_friends['User Name'].unique()
    print(unique_days_in_close_friends)
    print(unique_users_in_close_friends)
    print('Close Friends Group')
    for date in unique_days_in_close_friends:
        df_filter_by_date = df_close_friends[df_close_friends['Date'] == date]
        print('Date: ', date)
        for user in unique_users_in_close_friends:
            df_filter_by_date_by_user = df_filter_by_date[df_filter_by_date['User Name'] == user]
            print('\tUser:', user, ',\tNo. of Messages: ', len(df_filter_by_date_by_user))


if __name__ == '__main__':
    df = data_collection()
    df = pre_processing(df)
    df = text_pre_processing(df)

    # apply get sentiment score
    df['Sentiment_Score'] = df['Message'].apply(get_sentiment_score)
    print(df.head())

    populate_positive_negative_neutral_words()
    populate_most_used_words()
    populate_most_active_user()
    per_day_messages_by_each_memeber_in_each_group()
