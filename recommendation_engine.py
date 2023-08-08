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


    st.title("About: CitySeeker")
    
    intro_paragraph = (
    "Hello, everyone! I'm Chirag Saraogi, an immigrant from India with a decade of experience living in the United States. Having interacted with numerous fellow immigrants, I understand the challenges we face when searching for the perfect city to call home. "
    "To address this, I've developed CitySeeker, a sophisticated city recommendation system tailored to the needs of immigrants. It factors in essential considerations to help you find the ideal city for building your future.\n"
    )
    
    st.write(intro_paragraph)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("The Most Important Factors for Immigrants:")
    st.markdown("""
    When it comes to making a life-changing decision like choosing a city to live in, there are several key factors that play a significant role for immigrants. Here are the most critical aspects that my city recommendation system considers:
    - Job opportunities in the city
    - Cost of living
    - Immigrant community in the city
    - Domestic migration of people into the state each year
    - Weather year round
    - Safety record
    - Public education quality      
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Data Sources:")
    st.write("To create a robust and comprehensive recommendation system, we gathered data from reputable sources to compile a comprehensive data repository. The following are the primary data sources we used:")

    st.markdown("""
    - [Safety Record - WalletHub](https://wallethub.com/edu/safest-cities-in-america/41926)
    - [Cost of Living - Numbeo](https://www.numbeo.com/cost-of-living/region_rankings_current.jsp?region=021)
    - [Weather Year Round - Infoplease](https://www.infoplease.com/math-science/weather/climate-of-100-selected-us-cities)
    - [Job Opportunities in the City - Bureau of Labor Statistics](https://www.bls.gov/oes/current/oessrcma.htm)
    - [Countries of Origin for Immigrants - World Population Review](https://worldpopulationreview.com/country-rankings/us-immigration-by-country)
    - [State to State Domestic Migration by Year - U.S Census Bureau](https://www.census.gov/data/tables/time-series/demo/geographic-mobility/state-to-state-migration.html)
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Building the Recommendation System:")
    st.markdown ("""
    The following were the steps followed to make the Recommendation System:
    1. **Data Collection:** Gathered data from reliable sources mentioned above and preprocessed it, ensuring it is properly formatted for analysis.
    2. **Text Processing:** Performed **Part-of-Speech (POS) Tagging** and **Lemmatization** on profession data to create relevant tags for each city. To improve searchability, we use the **fasttext-wiki-news-subwords-300** model to find the top 5 similar words for each tag.
    3. **User Input Processing:** User inputs were processed to maintain consistent formatting during the matching process.
    4. **Similarity Calculation:** Used **Jaccard Similarity** method to calculate similarity scores between cities and user preferences, helping identify the best matches.
    5. **Recommendation Generation:** Based on the similarity scores and user preferences, the system filters and sorts the data to provide personalized city recommendations.
    6. **Web App Development:** The recommendation engine is hosted on a web app built using Streamlit, offering an intuitive and user-friendly interface for users to explore city options.
  
    Here's the link to the code and the data files used, [GitHub](https://github.com/chiragsaraogi15/City-Recommendations-for-Immigrants)
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Why only Immigrants:")
    
    st.markdown("""
    As we consider the factors that influence immigrant settlement patterns in the U.S., we find the following trends:
    - Immigrants often prefer to **live near their own communities** when relocating to a new country.
    - Approximately **74%** of immigrants are concentrated in just **six states: California, New York, Texas, Florida, New Jersey, and Illinois**. In contrast, **only 36%** of native-born individuals reside in these states.
    - Immigrant communities tend to gravitate toward areas with **major international airports**.
    - **Urban areas** are popular choices for immigrant settlement.

    Taking these insights into account, our recommendation system is designed to focus on the **38 major U.S. cities** that are commonly chosen by immigrants, rather than the hundreds of cities available. Additionally, the system considers the user's home country selection to provide tailored recommendations based on the existing **immigrant population** from that specific country in the chosen city.
    """
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

    

        
       
        
        
           