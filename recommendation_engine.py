import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import ast
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def get_pos_tag(token):
    pos_tag = nltk.pos_tag([token])[0][1]
    return pos_tag

# Function to tokenize, lemmatize with POS tagging, and remove stop words from a text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = []
    for token in tokens:
        pos_tag = get_pos_tag(token.lower())
        if token.lower() not in stop_words:
            if pos_tag.startswith('V'):  # Verb
                lemmatized_tokens.append(lemmatizer.lemmatize(token.lower(), pos='v'))
            elif pos_tag.startswith('N'):  # Noun
                lemmatized_tokens.append(lemmatizer.lemmatize(token.lower(), pos='n'))
            else:
                lemmatized_tokens.append(token.lower())
    return lemmatized_tokens


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def final_recommendations(user_profession, coldest_temp, hottest_temp, choice):   
    
    df = pd.read_pickle('tagged_data.pkl')

    selected_columns = ['SAFETY_INDEX', 'COST_OF_LIVING_INDEX', 'WINTER_COLDEST_TEMP',
                        'SUMMER_HOTTEST_TEMP', 'PROFESSION', 'TOT_EMP', 'H_MEAN', 'A_MEAN', 'TAGS','EXPANDED_TAGS',choice,'STATE_x','BEST_SUBURBS','IMAGE_LINK','DIFFERENT_STATE_MOVE_IN']
                    
    filtered_df = df[selected_columns]
      
    filtered_df = filtered_df[
        (filtered_df['WINTER_COLDEST_TEMP'] >= coldest_temp) &
        (filtered_df['SUMMER_HOTTEST_TEMP'] <= hottest_temp)
    ]

    user_input_tokens = preprocess_text(user_profession)
    user_input_tags = set(user_input_tokens)

    def string_to_set(s):
        try:
            elements = s.strip("[]").replace("'", "").split(', ')
            return set(elements)
        except (ValueError, SyntaxError):
            # If the literal_eval fails, return an empty set
            return set()
            

    filtered_df['EXPANDED_TAGS'] = filtered_df['EXPANDED_TAGS'].apply(string_to_set)

    filtered_df['Profession_Score'] = filtered_df['EXPANDED_TAGS'].apply(lambda x: jaccard_similarity(x, user_input_tags))

    filtered_df = filtered_df[filtered_df['Profession_Score'] > 0] \
                     .sort_values(by='Profession_Score', ascending=False) \
                     .groupby('CITY').head(1) \
                     .sort_index()

    filtered_df = filtered_df.sort_values(by=['TOT_EMP','A_MEAN',choice,'DIFFERENT_STATE_MOVE_IN','COST_OF_LIVING_INDEX','SAFETY_INDEX'],
                                          ascending=[False, False, False, False, True, False])
                                          
    max_top_cities = 2
    grouped_df = filtered_df.groupby('STATE_x').head(max_top_cities)
    mask = filtered_df.index.isin(grouped_df.index)
    filtered_df = filtered_df.loc[mask]
    
    
    desired_columns = ['STATE_x','PROFESSION', 'TOT_EMP', 'A_MEAN', choice, 'WINTER_COLDEST_TEMP', 'SUMMER_HOTTEST_TEMP','BEST_SUBURBS','IMAGE_LINK','DIFFERENT_STATE_MOVE_IN']
    filtered_df = filtered_df[desired_columns]
    
    
    return filtered_df

def final_recommendations_2(choice):   
    
    df_2 = pd.read_pickle('tagged_data.pkl')

    selected_columns_2 = ['SAFETY_INDEX', 'COST_OF_LIVING_INDEX', 'WINTER_COLDEST_TEMP',
                    'SUMMER_HOTTEST_TEMP',choice,'STATE_x','BEST_SUBURBS','IMAGE_LINK','DIFFERENT_STATE_MOVE_IN']
                    
    filtered_df_2 = df_2[selected_columns_2]
    
    filtered_df_2 = filtered_df_2.drop_duplicates()
    filtered_df_2 = filtered_df_2.sort_values(by = [choice, 'DIFFERENT_STATE_MOVE_IN','COST_OF_LIVING_INDEX', 'SAFETY_INDEX'], ascending = [False, False, True, False])
    
    return filtered_df_2


stop_words = set(stopwords.words('english'))

    

        
       
        
        
           