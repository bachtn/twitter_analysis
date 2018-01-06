import numpy as np
import pandas as pd
import json

### Feature extractor
def get_word_features(word, word_dictt, emoji_dict, word_def='hello', emoji_def='😂'):
    """ 
    The feature vector is composed of 2 parts: Word features and Emoji features
    If the word is an Emoji, than the 1st part of the vector is filled with 0
    	and the 2nd is filled with the Emoji features
    If the word is not an Emoji, than the 1st part is filled with the word features
    	and the 2nd is filled with 0.
	If the word does not exist, all the vector filled with 0

    @return numpy array with size = (nb_word_features + nb_emoji_features)
    """
    word_nb_features = len(word_dict[word_def])
    emoji_nb_features = len(emoji_dict[emoji_def])
    feature_array = np.zeros((word_nb_features + emoji_nb_features))
    if word in word_dict:
        # Fill the first part of the array with word features
        feature_array[:word_nb_features] = word_dict[word]
    elif word in emoji_dict:
        # Fill the second part of the array with emoji features
        feature_array[emoji_nb_features+1:] = emoji_dict[word]
    return feature_array

### Words
def get_word_sentiment_dict(dataset_path, delimiter='\t', class_column='word',
        remove_useless_columns=True, threshold=0):
    df = pd.read_csv(dataset_path, delimiter=delimiter)
    df[class_column] = df[class_column].str.lower()
    if remove_useless_columns:
        useful_columns = get_useful_columns(df, class_column=class_column, threshold=threshold)
        df = df[useful_columns].copy()
    df_dict = df.set_index(class_column).T.apply(tuple).to_dict()
    return df_dict

### Emojis
# Polarity and Sentiment dict
def get_emoji_dict(polarity_path, sentiment_path):
    polarity_dict = get_emoji_polarity_dict(polarity_path)
    sentiment_df = get_emoji_sentiment_df(sentiment_path)
    # Add new column for popularity with a default value 0
    sentiment_df['polarity'] = 0
    # Fill with the polarity values! (878 emojis with a missing polarity value!)
    for emoji, polarity in polarity_dict.items():
        sentiment_df.loc[sentiment_df['emoji'] == emoji, 'polarity'] = polarity
    # Convert to dict
    sentiment_dict = sentiment_df.set_index('emoji').T.apply(tuple).to_dict()
    return sentiment_dict

# Sentiment (Negative, Neutral, Positive)
def get_emoji_sentiment_df(sentiment_path):
    emoji_sentiment_df = pd.read_csv(sentiment_path, delimiter=',')
    # Convert scores to percentages between 0 and 1
    target_columns = ['negative', 'neutral', 'positive']
    for column in target_columns:
        emoji_sentiment_df[column] /= emoji_sentiment_df['occurrences']
    # Filter columns (remove useless ones)
    target_columns.append('emoji')
    emoji_sentiment_df = emoji_sentiment_df.filter(target_columns, axis=1)
    return emoji_sentiment_df

# Polarity (values between -4 and 4)
def get_emoji_polarity_dict(emoji_data_path):
    """Returns a dict with the emoji unicode as key and the emoji polarity as value"""
    with open(emoji_polarity_path) as json_data:
        emoji_polarities = json.load(json_data)
        
    emoji_polarity_dict = {}
    for emoji_val in emoji_polarities:
        emoji_polarity_dict[emoji_val["emoji"]] = emoji_val["polarity"]
    return emoji_polarity_dict

### Tools
def get_useful_columns(df, class_column='word', threshold=0):
    """Used to eliminate the columns that have many missing values
    If col_value = 0 -> missing_val = True
    """
    empty_columns = []
    for column_name in list(df.columns):
        if len(df.loc[df[column_name] != 0.]) <= threshold:
            empty_columns.append(column_name)
    useful_columns = list(set(df.columns) - set(empty_columns))
    if class_column not in useful_columns:
        useful_columns.append(class_column)
    print("Useful columns are:", useful_columns)
    return useful_columns


if __name__ == '__main__':
    # Word datasets path
    words_sentiment_path = './../tools/word/words.csv'

    # Emoji datasets path
    emoji_polarity_path = './../tools/emoji/emoji_emotion.json'
    emoji_sentiment_path = './../tools/emoji/emoji_sentiments.csv'
    # Get feature dicts
    word_dict = get_word_sentiment_dict(dataset_path=words_sentiment_path, threshold=3500)
    emoji_dict = get_emoji_dict(emoji_polarity_path, emoji_sentiment_path)
    # Test
    print(get_word_features('fun', word_dict, emoji_dict))
    print(get_word_features('#fun', word_dict, emoji_dict))
    print(get_word_features('😂', word_dict, emoji_dict))
