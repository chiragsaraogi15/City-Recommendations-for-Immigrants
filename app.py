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

    filtered_df = filtered_df.sort_values(by=['TOT_EMP', 'A_MEAN','COST_OF_LIVING_INDEX', choice, 'SAFETY_INDEX'],
                                          ascending=[False, False, True, False, False])

    desired_columns = ['PROFESSION', 'TOT_EMP', 'A_MEAN', choice, 'WINTER_COLDEST_TEMP', 'SUMMER_HOTTEST_TEMP']
    filtered_df = filtered_df[desired_columns]
    return filtered_df


# Your remaining code for the recommendations model
stop_words = set(stopwords.words('english'))

# heading and description text 
st.title("USA City Recommendation System")

long_paragraph = (
        "United States of America, a dream country for many immigrants coming from different parts of the world are often overwhelmed by the number of opportunities and amazing cities to choose from. "
        "Some immigrants move here to be with their families, and some move here to make a better life and find their dream job. \n"       
)
st.write(long_paragraph)

long_paragraph2 = (
        "USA is big country comprising of 50 states and 100's of large cities, and deciding where to live can be extremely challenging. "
        "As immigrants, factors like **job availability**, **community**, **weather**, **education**, **affordability** and **safety** are a few factors that are important when making a decision on where to live. \n"
)

st.write(long_paragraph2)

st.markdown("I have created a system that takes in your preferences and provides you with recommendations for cities to consider living in. Let's begin by answering 4 simple questions about you. \n")

# Step 1: Take input for user's profession name

st.markdown("<h3>What kind of work do you do?</h3>", unsafe_allow_html=True)
user_profession = st.text_input("")


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
st.subheader("How do you like your winters?")
# Add a blank default choice as the first item in the list
winter_temp_choices = ["Select an option"] + list(winter_temp_mapping.keys())
selected_winter_temp = st.selectbox("", winter_temp_choices)

# Get the actual coldest temperature based on the user's choice
if selected_winter_temp != "Select an option":
    coldest_temp = winter_temp_mapping[selected_winter_temp]
else:
    coldest_temp = None
    

# Step 3: Take input for hottest temperature comfortable with
st.subheader("How do you like your summers?")
# Add a blank default choice as the first item in the list
summer_temp_choices = ["Select an option"] + list(summer_temp_mapping.keys())
selected_summer_temp = st.selectbox("", summer_temp_choices)

# Get the actual hottest temperature based on the user's choice
if selected_summer_temp != "Select an option":
    hottest_temp = summer_temp_mapping[selected_summer_temp]
else:
    hottest_temp = None
 
 
# Step 4: Take input for the country from the user
countries = ['Bangladesh', 'Brazil', 'Canada', 'China', 'Colombia', 'Cuba', 'Dominican Republic', 'El Salvador',
             'Ethiopia', 'Guatemala', 'Haiti', 'India', 'Iran', 'Iraq', 'Jamaica', 'Korea', 'Mexico', 'Nepal',
             'Nigeria', 'Pakistan', 'Philippines', 'United Kingdom', 'Vietnam', 'Other Countries']



# User Input: Country Selection
st.subheader("Which country are you originally from?")
countries_choices = ["Select an option"] + countries
index_choice = st.selectbox("", range(len(countries_choices)), format_func=lambda i: countries_choices[i])
choice = countries[index_choice - 1]


if st.button("Submit"):
    recommendations_df = final_recommendations(user_profession, coldest_temp, hottest_temp, choice)
   
    column_mapping = {
       'CITY': 'CITY',
       'PROFESSION': 'PROFESSION',
       'TOT_EMP': 'EMPLOYMENT COUNT',
       'A_MEAN': 'AVERAGE ANNUAL SALARY',
       choice: 'IMMIGRANT COUNT',
       'WINTER_COLDEST_TEMP': 'COLDEST TEMPERATURE IN WINTERS',
       'SUMMER_HOTTEST_TEMP': 'HOTTEST TEMPERATURE IN SUMMERS'
    }
   
    recommendations_df = recommendations_df.rename(columns=column_mapping)
   
    st.subheader("Top 5 Recommendations:")
   
    # Create a CSS style for the cards
    card_style = """
    <style>
    .card {
        padding: 1rem;
        margin: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        background-color: #f9f9f9;
    }
    </style>
    """

    st.write(card_style, unsafe_allow_html=True)

    # Display each row as a card
    for index, row in recommendations_df.head(5).iterrows():
        st.write(
            f'<div class="card"><h2>{index}</h2>'
            f'<p><strong>PROFESSION:</strong> {row["PROFESSION"]}</p>'
            f'<p><strong>EMPLOYMENT COUNT:</strong> {row["EMPLOYMENT COUNT"]}</p>'
            f'<p><strong>AVERAGE ANNUAL SALARY:</strong> {row["AVERAGE ANNUAL SALARY"]}</p>'
            f'<p><strong>IMMIGRANT COUNT:</strong> {row["IMMIGRANT COUNT"]}</p>'
            f'<p><strong>COLDEST TEMPERATURE IN WINTERS:</strong> {row["COLDEST TEMPERATURE IN WINTERS"]}</p>'
            f'<p><strong>HOTTEST TEMPERATURE IN SUMMERS:</strong> {row["HOTTEST TEMPERATURE IN SUMMERS"]}</p></div>',
            unsafe_allow_html=True,
        )




