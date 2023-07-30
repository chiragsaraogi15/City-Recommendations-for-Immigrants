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
                        'SUMMER_HOTTEST_TEMP', 'PROFESSION', 'TOT_EMP', 'H_MEAN', 'A_MEAN', 'TAGS','EXPANDED_TAGS',choice,'STATE_x','BEST_SUBURBS','IMAGE_LINK']

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

    desired_columns = ['STATE_x','PROFESSION', 'TOT_EMP', 'A_MEAN', choice, 'WINTER_COLDEST_TEMP', 'SUMMER_HOTTEST_TEMP','BEST_SUBURBS','IMAGE_LINK']
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
st.markdown("<h4 style='margin: 0;'>What kind of work do you do?</h4>", unsafe_allow_html=True)
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
st.markdown("<h4 style='margin: 0;'>How do you like your winters?</h4>", unsafe_allow_html=True)
# Add a blank default choice as the first item in the list
winter_temp_choices = ["Select an option"] + list(winter_temp_mapping.keys())
selected_winter_temp = st.selectbox("", winter_temp_choices)

# Get the actual coldest temperature based on the user's choice
if selected_winter_temp != "Select an option":
    coldest_temp = winter_temp_mapping[selected_winter_temp]
else:
    coldest_temp = None
    

# Step 3: Take input for hottest temperature comfortable with
st.markdown("<h4 style='margin: 0;'>How do you like your summers?</h4>", unsafe_allow_html=True)
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
st.markdown("<h4 style='margin: 0;'>Which country are you originally from?</h4>", unsafe_allow_html=True)
countries_choices = ["Select an option"] + countries
index_choice = st.selectbox("", range(len(countries_choices)), format_func=lambda i: countries_choices[i])
choice = countries[index_choice - 1]


if st.button("Submit"):
    recommendations_df = final_recommendations(user_profession, coldest_temp, hottest_temp, choice)
   
    column_mapping = {
       'CITY': 'City',
       'PROFESSION': 'Profession',
       'TOT_EMP': 'Employment Count',
       'A_MEAN': 'Average Annual Salary',
       choice: 'Immigrant Count',
       'WINTER_COLDEST_TEMP': 'Coldest Temperature in Winters',
       'SUMMER_HOTTEST_TEMP': 'Hottest Temperature in Summers',
       'STATE_x': 'State',
       'BEST_SUBURBS': 'Best Suburbs',
       'IMAGE_LINK': 'Image Link'
    }
    
   
    recommendations_df = recommendations_df.rename(columns=column_mapping)
   
    st.subheader("Top 5 Recommendations:")
   
    
    card_style = """
    <style>
    .card {
        display: flex;
        padding: 1rem;
        margin: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        background-color: #f9f9f9;
    }

    .card-left {
        flex: 1;
    }

    .card-right {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .card-img-container {
        width: 180px;
        height: 180px;
        overflow: hidden;
    }

    .card-img {
        width: 90%;
        height: 90%;
        object-fit: cover;
        border: 4px solid #ccc;
    }
    </style>
    """

    st.write(card_style, unsafe_allow_html=True)

    # Display each row as a card with city number and image
    for idx, (index, row) in enumerate(recommendations_df.head(5).iterrows(), 1):
        employment_count = f'{row["Employment Count"]:,}'
        average_salary = f'${row["Average Annual Salary"]:,.0f}'  # Remove decimal places
        immigrant_count = f'{row["Immigrant Count"]:,}'
        coldest_temp_winter = f'{round(row["Coldest Temperature in Winters"]):,}'  # Round and remove decimals
        hottest_temp_summer = f'{round(row["Hottest Temperature in Summers"]):,}'  # Round and remove decimals

        best_suburbs_value = row["Best Suburbs"]
        best_suburbs_link = f'<a href="{best_suburbs_value}" target="_blank">Click for best suburbs of the city</a>'

        image_link = row["Image Link"]
        image_html = f'<img class="card-img" src="{image_link}" alt="{index} Image">'

        city_card_html = (
            f'<div class="card"><div class="card-left"><h2>{idx}. {index}</h2>'
            f'<p><strong>Profession:</strong> {row["Profession"]}</p>'
            f'<p><strong>Employment Count:</strong> {employment_count}</p>'
            f'<p><strong>Average Annual Salary:</strong> {average_salary}</p>'
            f'<p><strong>Immigrant Count:</strong> {immigrant_count}</p>'
            f'<p><strong>Coldest Temperature in Winters:</strong> {coldest_temp_winter}°F</p>'
            f'<p><strong>Hottest Temperature in Summers:</strong> {hottest_temp_summer}°F</p>'
            f'<p>{best_suburbs_link}</p></div>'
            f'<div class="card-right">{image_html}</div></div>'
        )

        st.write(city_card_html, unsafe_allow_html=True)