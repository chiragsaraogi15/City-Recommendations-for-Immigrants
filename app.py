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

def collect_feedback(feedback_text):
    if feedback_text:
        feedback_df = pd.DataFrame({'Feedback': [feedback_text]})
        feedback_df.to_csv('feedback.csv', mode='a', index=False, header=not st.session_state.feedback_saved)
        st.session_state.feedback_saved = True
        st.success("Thank you for your feedback!")

def about_page():
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
    - Job Opportunities in the City
    - Cost of Living
    - Immigrant Community in the City
    - Weather Year Round
    - Safety Record
    - Public Education Quality      
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
    """)
    
     
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
            

    filtered_df['EXPANDED_TAGS'] = filtered_df['EXPANDED_TAGS'].apply(string_to_set)

    filtered_df['Profession_Score'] = filtered_df['EXPANDED_TAGS'].apply(lambda x: jaccard_similarity(x, user_input_tags))

    filtered_df = filtered_df[filtered_df['Profession_Score'] > 0] \
                     .sort_values(by='Profession_Score', ascending=False) \
                     .groupby('CITY').head(1) \
                     .sort_index()

    filtered_df = filtered_df.sort_values(by=['TOT_EMP', 'A_MEAN','COST_OF_LIVING_INDEX', choice, 'SAFETY_INDEX'],
                                          ascending=[False, False, True, False, False])
                                          
    max_top_cities = 2
    grouped_df = filtered_df.groupby('STATE_x').head(max_top_cities)
    mask = filtered_df.index.isin(grouped_df.index)
    filtered_df = filtered_df.loc[mask]
    
    
    desired_columns = ['STATE_x','PROFESSION', 'TOT_EMP', 'A_MEAN', choice, 'WINTER_COLDEST_TEMP', 'SUMMER_HOTTEST_TEMP','BEST_SUBURBS','IMAGE_LINK']
    filtered_df = filtered_df[desired_columns]
    
    
    return filtered_df

def final_recommendations_2(choice):   
    
    df_2 = pd.read_pickle('tagged_data.pkl')

    selected_columns_2 = ['SAFETY_INDEX', 'COST_OF_LIVING_INDEX', 'WINTER_COLDEST_TEMP',
                    'SUMMER_HOTTEST_TEMP',choice,'STATE_x','BEST_SUBURBS','IMAGE_LINK']
                    
    filtered_df_2 = df_2[selected_columns_2]
    
    filtered_df_2 = filtered_df_2.drop_duplicates()
    filtered_df_2 = filtered_df_2.sort_values(by = [choice, 'COST_OF_LIVING_INDEX', 'SAFETY_INDEX'], ascending = [False, True, False])
    
    return filtered_df_2


# Your remaining code for the recommendations model
stop_words = set(stopwords.words('english'))

def main():
    
    st.markdown(
        """
        <style>
        /* Remove the default Streamlit sidebar style */
        .sidebar .sidebar-content {
            width: 300px;
            padding: 2rem;
            background-color: #f9f9f9;
        }

        /* Style the sidebar navigation links */
        .sidebar .sidebar-list {
            list-style-type: none;
            padding: 0;
            margin: 0;
        }

        .sidebar .sidebar-list li {
            margin-bottom: 1rem;
        }

        .sidebar .sidebar-list li a {
            display: block;
            color: #333;
            text-decoration: none;
            font-size: 20px;
            padding: 10px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }

        .sidebar .sidebar-list li a:hover {
            background-color: #0575E6;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "About"))

    if page == "Home":
        # heading and description text 
        st.title("CitySeeker: Helping Immigrants find their Perfect City")

        intro_paragraph = (
            "Welcome to CitySeeker! The USA, a dream country for immigrants worldwide, offers endless opportunities and incredible cities to explore. We understand that choosing the right city can be both exciting and overwhelming. "
            "Whether you're reuniting with family, seeking better opportunities, or pursuing your dream career, we're here to help you find the ideal place to call home.\n"
        )
        st.write(intro_paragraph)

        long_paragraph2 = (
            "Choosing where to live in the USA can be overwhelming. As an immigrant, important factors like job opportunities, community, weather, education, affordability, and safety matter most.\n"
            )

        st.write(long_paragraph2)

        st.markdown("Discover your ideal city with our advanced recommendation system. Answer four simple questions, and we'll provide personalized city suggestions tailored to your preferences. So let's get started!")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Step 1: Take input for user's profession name
        st.markdown("<h4 style='margin: 0;'>What kind of work do you do?</h4>", unsafe_allow_html=True)
        st.write("Please be specific for better results. For example: Software Engineer, Doctor, Teacher, etc.")
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
            recommendations_df_2 = final_recommendations_2(choice)
            
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
            
            column_mapping_2 = {
               'CITY': 'City',
               choice: 'Immigrant Count',
               'WINTER_COLDEST_TEMP': 'Coldest Temperature in Winters',
               'SUMMER_HOTTEST_TEMP': 'Hottest Temperature in Summers',
               'STATE_x': 'State',
               'BEST_SUBURBS': 'Best Suburbs',
               'IMAGE_LINK': 'Image Link'
            }
            
           
            recommendations_df = recommendations_df.rename(columns=column_mapping)
            recommendations_df_2 = recommendations_df_2.rename(columns=column_mapping_2)
            
            len_df = len(recommendations_df)
            len_df_2 = 5-len_df
            
            if len_df >= 5:
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.subheader("Top Recommendations for your Preferences")
                
                # CSS style for the cards
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
                    border: 1px solid black;
                }

                @media screen and (max-width: 768px) {
                    /* Mobile layout */
                    .card {
                        flex-direction: column;
                        align-items: center;
                    }
                    .card-left, .card-right {
                        width: 100%;
                        text-align: center;
                    }
                    .card-right {
                        margin-top: 1rem;
                    }
                }
                </style>
                """

                st.write(card_style, unsafe_allow_html=True)

                # Display each row as a card with city number and image
                for idx, (index, row) in enumerate(recommendations_df.head(5).iterrows(), 1):
                    employment_count = f'{row["Employment Count"]:,}'
                    average_salary = f'${row["Average Annual Salary"]:,.0f}' if row["Average Annual Salary"] > 0 else "Not Available"
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
                        f'<p><strong>Home Country Immigrant Count:</strong> {immigrant_count}</p>'
                        f'<p><strong>Coldest Temperature in Winters:</strong> {coldest_temp_winter}°F</p>'
                        f'<p><strong>Hottest Temperature in Summers:</strong> {hottest_temp_summer}°F</p>'
                        f'<p>{best_suburbs_link}</p></div>'
                        f'<div class="card-right">{image_html}</div></div>'
                    )

                    st.write(city_card_html, unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
            
                message = """
                <font size="4"><b>Note:</b> this recommendation system takes into account factors like <b>job availabilities</b> for your profession, <b>average annual salary</b>, <b>cost of living</b>, <b>immigrant count</b> from your home country, and <b>safety index</b> in the city.</font>
                <font size="4">Cities have been ranked based on these factors and in the same order mentioned above.</font>
                """
                st.markdown(message, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                
            elif len_df == 0:
            
                st.markdown("<br>", unsafe_allow_html=True)
                
                message = """
                <font size="4"><b>Note:</b> Based on your selections above, the recommendation system could <b>not find</b> any cities matching your preferences.</font>
                <font size="4">Please try again by being more <b>specific about your profession</b> and <b>selecting other weather options</b>. </font>
                <font size="4">Also, below are the <b>top 5 cities occupied by immigrants from your home country</b>.</i></font>
                """
                st.markdown(message, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.subheader("Popular Cities")
                
                # CSS style for the cards
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

                .card-content {
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    flex: 3; /* Adjust this value to control the spacing */
                    margin-right: 1rem; /* Add some margin between the content and the image */
                }

                .card-img-container {
                    width: 180px;
                    height: 180px;
                    overflow: hidden;
                }

                .card-img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    border: 1px solid black;
                }

                @media screen and (max-width: 768px) {
                    /* Mobile layout */
                    .card {
                        flex-direction: column;
                        align-items: center;
                    }
                    .card-left, .card-right {
                        width: 100%;
                        text-align: center;
                    }
                    .card-right {
                        margin-top: 1rem;
                    }
                }
                </style>
                """

                st.write(card_style, unsafe_allow_html=True)

                # Display each row as a card with city number and image
                for idx, (index, row) in enumerate(recommendations_df_2.head(5).iterrows(), 1):
                    immigrant_count = f'{row["Immigrant Count"]:,}'
                    coldest_temp_winter = f'{round(row["Coldest Temperature in Winters"]):,}'  # Round and remove decimals
                    hottest_temp_summer = f'{round(row["Hottest Temperature in Summers"]):,}'  # Round and remove decimals

                    best_suburbs_value = row["Best Suburbs"]
                    best_suburbs_link = f'<a href="{best_suburbs_value}" target="_blank">Click for best suburbs of the city</a>'

                    image_link = row["Image Link"]
                    image_html = f'<img class="card-img" src="{image_link}" alt="{index} Image">'

                    city_card_html = (
                        f'<div class="card"><div class="card-left"><h2>{idx}. {index}</h2>'
                        f'<p><strong>Home Country Immigrant Count:</strong> {immigrant_count}</p>'
                        f'<p><strong>Coldest Temperature in Winters:</strong> {coldest_temp_winter}°F</p>'
                        f'<p><strong>Hottest Temperature in Summers:</strong> {hottest_temp_summer}°F</p>'
                        f'<p>{best_suburbs_link}</p></div>'
                        f'<div class="card-right">{image_html}</div></div>'
                    )

                    st.write(city_card_html, unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)

                
            else:  
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.subheader("Top Recommendations for your Preferences")
                
                # CSS style for the cards
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
                    border: 1px solid black;
                }

                @media screen and (max-width: 768px) {
                    /* Mobile layout */
                    .card {
                        flex-direction: column;
                        align-items: center;
                    }
                    .card-left, .card-right {
                        width: 100%;
                        text-align: center;
                    }
                    .card-right {
                        margin-top: 1rem;
                    }
                }
                </style>
                """

                st.write(card_style, unsafe_allow_html=True)

                # Display each row as a card with city number and image
                for idx, (index, row) in enumerate(recommendations_df.head(len_df).iterrows(), 1):
                    employment_count = f'{row["Employment Count"]:,}'
                    average_salary = f'${row["Average Annual Salary"]:,.0f}' if row["Average Annual Salary"] > 0 else "Not Available"
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
                        f'<p><strong>Home Country Immigrant Count:</strong> {immigrant_count}</p>'
                        f'<p><strong>Coldest Temperature in Winters:</strong> {coldest_temp_winter}°F</p>'
                        f'<p><strong>Hottest Temperature in Summers:</strong> {hottest_temp_summer}°F</p>'
                        f'<p>{best_suburbs_link}</p></div>'
                        f'<div class="card-right">{image_html}</div></div>'
                    )

                    st.write(city_card_html, unsafe_allow_html=True)
                    
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                message = """
                <font size="4"><b>Note:</b> Recommendations provided above are cities that matched your preferences.\n</font>
                <font size="4">To get more cities matching your preferences, please try again by being more specific about your profession and selecting other weather options. </font>
                <font size="4">Also, below are other top cities occupied by immigrants from your home country.</font>
                """
                st.markdown(message, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.subheader("Other Popular Cities")
                
                existing_indexes = set(recommendations_df.index)
                
                cards_printed = 0
                
                # CSS style for the cards
                card_style_2 = """
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

                .card-content {
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    flex: 3; /* Adjust this value to control the spacing */
                    margin-right: 1rem; /* Add some margin between the content and the image */
                }

                .card-img-container {
                    width: 180px;
                    height: 180px;
                    overflow: hidden;
                }

                .card-img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    border: 1px solid black;
                }

                @media screen and (max-width: 768px) {
                    /* Mobile layout */
                    .card {
                        flex-direction: column;
                        align-items: center;
                    }
                    .card-left, .card-right {
                        width: 100%;
                        text-align: center;
                    }
                    .card-right {
                        margin-top: 1rem;
                    }
                }
                </style>
                """
                
                st.write(card_style_2, unsafe_allow_html=True)
                
                # Display each row as a card with city number and image
                for idx, (index, row) in enumerate(recommendations_df_2.iterrows(), 1):
                    # Check if the index is not present in recommendations_df
                    if index not in existing_indexes:
                        immigrant_count = f'{row["Immigrant Count"]:,}'
                        coldest_temp_winter = f'{round(row["Coldest Temperature in Winters"]):,}'  # Round and remove decimals
                        hottest_temp_summer = f'{round(row["Hottest Temperature in Summers"]):,}'  # Round and remove decimals

                        best_suburbs_value = row["Best Suburbs"]
                        best_suburbs_link = f'<a href="{best_suburbs_value}" target="_blank">Click for best suburbs of the city</a>'

                        image_link = row["Image Link"]
                        image_html = f'<img class="card-img" src="{image_link}" alt="{index} Image">'

                        city_card_html = (
                            f'<div class="card"><div class="card-left"><h2>{idx}. {index}</h2>'
                            f'<p><strong>Home Country Immigrant Count:</strong> {immigrant_count}</p>'
                            f'<p><strong>Coldest Temperature in Winters:</strong> {coldest_temp_winter}°F</p>'
                            f'<p><strong>Hottest Temperature in Summers:</strong> {hottest_temp_summer}°F</p>'
                            f'<p>{best_suburbs_link}</p></div>'
                            f'<div class="card-right">{image_html}</div></div>'
                        )

                        st.write(city_card_html, unsafe_allow_html=True)

                        # Increment the counter for the number of cards printed
                        cards_printed += 1

                        # Check if we have printed enough cards (reached len_df_2)
                        if cards_printed >= len_df_2:
                            break
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
            st.subheader("Feedback")
            feedback_text = st.text_area("Please share your feedback with us:", max_chars=1000)
            if st.button("Submit Feedback"):
                collect_feedback(feedback_text)
    
    elif page == "About":
        about_page()
        
if __name__ == "__main__":
    main()
