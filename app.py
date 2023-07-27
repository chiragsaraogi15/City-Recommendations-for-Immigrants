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
                        'SUMMER_HOTTEST_TEMP', 'PROFESSION', 'TOT_EMP', 'H_MEAN', 'A_MEAN', 'TAGS','EXPANDED_TAGS',choice]

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


    filtered_df['TAGS'] = filtered_df['TAGS'].apply(string_to_set)

    filtered_df['EXPANDED_TAGS'] = filtered_df['EXPANDED_TAGS'].apply(string_to_set)

    filtered_df['Profession_Score'] = filtered_df['EXPANDED_TAGS'].apply(lambda x: jaccard_similarity(x, user_input_tags))

    filtered_df = filtered_df.sort_values(by='Profession_Score', ascending=False) \
                             .groupby('CITY').head(1) \
                             .sort_index()

    filtered_df = filtered_df.sort_values(by=['TOT_EMP', 'A_MEAN', choice, 'SAFETY_INDEX', 'COST_OF_LIVING_INDEX'],
                                          ascending=[False, False, False, False, True])

    desired_columns = ['PROFESSION', 'TOT_EMP', 'A_MEAN', choice, 'WINTER_COLDEST_TEMP', 'SUMMER_HOTTEST_TEMP']
    filtered_df = filtered_df[desired_columns]
    return filtered_df


# Your remaining code for the recommendations model
stop_words = set(stopwords.words('english'))

# heading and description text 
st.markdown("# Welcome to My Immigrant USA City Recommendations Website")
st.markdown("United States of America, a dream country for many immigrants coming from different parts of the world are often overwhelmed by the number of opportunities and amazing cities to choose from.")

# Step 1: Take input for user's profession name
user_profession = st.text_input("Enter your profession name:")

# Step 2: taking weather input preferences
coldest_temp = None
hottest_temp = None

winter_temp_mapping = {
    "1. Not cold at all (above 60°F)": 60,
    "2. Slightly cool (50-60°F)": 50,
    "3. Little cold (40-50°F)": 40,
    "4. Very cold (30-40°F)": 30,
    "5. Extremely cold (below 30°F)": 20
}

summer_temp_mapping = {
    "1. Not hot at all (below 65°F)": 65,
    "2. Slightly hot (65 - 70°F)": 70,
    "3. Little hot (70 - 80°F)": 80,
    "4. Very hot (80 - 90°F)": 90,
    "5. Extreme hot (above 90°F)": 100
}

# Step 2: Take input for coldest temperature comfortable with
st.subheader("How do you prefer your winters to be?")
# Add a blank default choice as the first item in the list
winter_temp_choices = ["Select an option"] + list(winter_temp_mapping.keys())
coldest_temp = st.selectbox(winter_temp_choices)

# Step 3: Take input for hottest temperature comfortable with
st.subheader("How do you prefer your summers to be?")
# Add a blank default choice as the first item in the list
summer_temp_choices = ["Select an option"] + list(summer_temp_mapping.keys())
hottest_temp = st.selectbox(summer_temp_choices)

if coldest_temp != "Select an option" and hottest_temp != "Select an option":
    coldest_temp = float(coldest_temp.split()[0])
    hottest_temp = float(hottest_temp.split()[0])
 
# Step 4: Take input for the country from the user
countries = ['Mexico', 'China', 'India', 'Philippines', 'Dominican Republic', 'Cuba', 'Vietnam', 'El Salvador',
             'Korea', 'Jamaica', 'Brazil', 'Haiti', 'Colombia', 'Pakistan', 'Iraq', 'Bangladesh', 'Nigeria',
             'Ethiopia', 'Canada', 'Iran', 'Guatemala', 'United Kingdom', 'Nepal', 'Other Countries']

# User Input: Country Selection
st.subheader("Choose your country from the list:")
countries_choices = ["Select an option"] + countries
index_choice = st.selectbox(range(len(countries_choices)), format_func=lambda i: countries_choices[i])
choice = countries[index_choice - 1]


if st.button("Submit"):
   recommendations_df = final_recommendations(user_profession, coldest_temp, hottest_temp, choice)
   st.subheader("Top 5 Recommendations:")
   st.dataframe(recommendations_df.head(5))    
    
