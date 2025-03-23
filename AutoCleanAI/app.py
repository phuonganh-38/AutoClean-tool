import streamlit as st
from tool import Tool, CorrectType, Visualisation
from load_data import LoadData

import re

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&display=swap');

    /* Background */
    html, body, [data-testid="stApp"] {
        background: linear-gradient(to bottom, #b5e4f6, white);
        font-family: "Inter", sans-serif;
        color: #1d1d1f;
    }
    
    /* Header */
    div[data-testid="stAppViewContainer"] h1 {
        font-family: "Inter", sans-serif;
        font-size: 30px;
        font-weight: 500;
        color: black;
        text-align: center;
    }

    /* Subheader */
    .slogan {
        font-family: "Inter", sans-serif;
        font-size: 50px !important;
        font-weight: 700;
        background: linear-gradient(to right, #022640, #5786AB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        margin-top: -30px !important;
    }
    

    /* Button */
    .stButton > button {
        font-family: "Inter", sans-serif;
        background-color: DodgerBlue;
        color: white;
        font-size: 14px;
        letter-spacing: -0.022em;
        border: 1px;
        border-radius: 600px;
        padding: 6px 15px;
    }

    /* Header sections */
    .feature-section {
        font-family: 'Inter', sans-serif;
        font-size: 28px;
        font-weight: 600;
        background: linear-gradient(to right, #022640, #5786AB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        margin-top: -34px;
        margin-bottom: 40px;
    }

    .data-section {
        font-family: 'Inter', sans-serif;
        font-size: 28px;
        font-weight: 600;
        background: linear-gradient(to right, #022640, #5786AB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        margin-top: 50px;
        margin-bottom: -38px;
    }

    .data-preview-section {
        font-family: 'Inter', sans-serif;
        font-size: 28px;
        font-weight: 600;
        background: linear-gradient(to right, #022640, #5786AB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        margin-top: -34px;
        margin-bottom: 10px;
    }

    .explore-section {
        font-family: 'Inter', sans-serif;
        font-size: 28px;
        font-weight: 600;
        background: linear-gradient(to right, #022640, #5786AB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        margin-top: 40px;
        margin-bottom: -30px;
    }

    .clean-section {
        font-family: 'Inter', sans-serif;
        font-size: 28px;
        font-weight: 600;
        background: linear-gradient(to right, #022640, #5786AB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        margin-top: 40px;
        margin-bottom: 10px;
    }

    .subheader {
        font-size: 16px !important;
        letter-spacing: 0.05em !important;
        line-height: 1em !important;
        margin-top: 35px !important;
        margin-bottom: 20px !important;
    }

    /* Feature section */
    .feature-subheader {
        font-size: 22px !important;
        font-weight: 600;
        color: black;
        line-height: 1.3em !important;
        margin-bottom: 20px;
    }

    .feature-content {
        font-size: 18px !important;
        font-weight: 500;
        color: #1d1d1f;
        line-height: 1.3em !important;
    }

    /* Text font */
    .question-1 {
        font-family: 'Inter', sans-serif;
        font-size: 16px !important;
        font-weight: 500;
        color: #1d1d1f;
        margin-top: 10px;
        margin-bottom: -50px !important;
    }

    .extract {
        font-family: 'Inter', sans-serif;
        font-size: 16px !important;
        font-weight: 500;
        color: #1d1d1f;
        margin-top: 20px;
        margin-bottom: -110px !important;
    }

    .plot-instruction {
        font-family: 'Inter', sans-serif;
        font-size: 16px !important;
        font-weight: 500;
        color: #1d1d1f;
        margin-top: 18px;
        margin-bottom: -82px !important;
    }

    .cleaning-option {
        font-family: 'Inter', sans-serif;
        font-size: 16px !important;
        font-weight: 500;
        color: #1d1d1f;
        margin-top: 16px;
        margin-bottom: -100px !important;
    }
    
    /* Radio */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        background-color: #1d1d1f; 
        padding: 12px;
        border-radius: 5px;
        margin-top: -10px;
    }
    
    /* Checkbox */
    div[data-testid="stCheckbox"] label {
        font-family: 'Inter', sans-serif !important;
        font-size: 16px !important;
        font-weight: 400 !important;
        color: #1d1d1f !important;
    }

    div[data-testid="stCheckbox"] label {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[data-testid="stCheckbox"] {
        background-color: #1d1d1f;
        padding: 8px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stAlert {
        background-color: black !important;
        border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div.stDownloadButton > button {
        background-color: DodgerBlue !important;
        color: white !important; 
        border-radius: 50px !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
    }

    div.stDownloadButton > button:hover {
        background-color: #ff0000 !important;
        border: 2px solid #cc0000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



class App:
    def __init__(self):
        self.df = None

    def run(self):
        st.title("AutoClean AI")
        st.markdown('<div style="text-align: center;"><h2 class="slogan">Clean faster. Effort less.</h2></div>',
        unsafe_allow_html=True
        )

        # Features-section
        st.markdown('<div> <h3 class="subheader">FEATURES</h3></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-section">Here is what you get with AutoClean AI</div>',unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3, gap='large')
        st.divider() # Row-gap
        col4, col5, col6 = st.columns(3, gap='large')
        st.divider() # Row-gap
        col7, col8, col9 = st.columns(3, gap='large')

        # Feature 1: Smart data type recognition
        with col1:
            st.image("https://img.icons8.com/?size=100&id=12640&format=png&color=339AF0", width=60)
            st.markdown('<div class="feature-subheader" style="white-space: nowrap;">Smart data type recognition</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-content">Automatically identifies numerical and categorical data for further analysis.</div>', unsafe_allow_html=True)
        
        # Feature 2: Intelligent number extraction
        with col2:
            st.image("https://img.icons8.com/?size=100&id=ZhO44hs9Llh4&format=png&color=339AF0", width=60) 
            st.markdown('<div class="feature-subheader" style="white-space: nowrap;">Intelligent number extraction</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-content">Extracts numeric values from messy text columns, converting them into usable formats.</div>', unsafe_allow_html=True)

        # Feature 3: Auto-generated visualisations
        with col3:
            st.image("https://img.icons8.com/?size=100&id=3005&format=png&color=339AF0", width=60)
            st.markdown('<div class="feature-subheader" style="white-space: nowrap;">Auto-generated visualisations</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-content">Instantly generates the most suitable charts based on your dataset, no manual setup needed.</div>', unsafe_allow_html=True)

        # Feature 4: Standardized formatting
        with col4:
            st.image("https://img.icons8.com/?size=100&id=3005&format=png&color=339AF0", width=60)
            st.markdown('<div class="feature-subheader" style="white-space: nowrap;">Standardized formatting</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-content">Trims excess spaces and standardizes text formatting.</div>', unsafe_allow_html=True)       
        
        
        # Feature 5: Intelligent text correction
        with col5:
            st.image("https://img.icons8.com/?size=100&id=50836&format=png&color=339AF0", width=60)
            st.markdown('<div class="feature-subheader" style="white-space: nowrap;">Intelligent text correction</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-content">Fixes typos, merges similar text values, and standardizes categorical data with Fuzzy Matching.</div>', unsafe_allow_html=True)
        
        # Feature 6: Missing value handling
        with col6:
            st.image("https://img.icons8.com/?size=100&id=21949&format=png&color=339AF0", width=60)
            st.markdown('<div class="feature-subheader" style="white-space: nowrap;">Missing value handling</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-content">Detects and fills missing values intelligently using mean, median, or categorical imputation.</div>', unsafe_allow_html=True)
        
        # Feature 7: Duplicate data removal
        with col7:
            st.image("https://img.icons8.com/?size=100&id=98618&format=png&color=339AF0", width=60)
            st.markdown('<div class="feature-subheader" style="white-space: nowrap;">Duplicate data removal</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-content">Automatically detects and removes duplicate records to keep your data clean.</div>', unsafe_allow_html=True)
        
        # Feature 8: Outlier detection
        with col8:
            st.image("https://img.icons8.com/?size=100&id=1724&format=png&color=339AF0", width=60)
            st.markdown('<div class="feature-subheader" style="white-space: nowrap;">Outlier detection</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-content">Automatically detects and removes anomalies using Isolation Forest.</div>', unsafe_allow_html=True)

        # Initialize session state
        if "df" not in st.session_state:
            st.session_state.df = None
        if "uploaded_file_name" not in st.session_state:
            st.session_state.uploaded_file_name = None
        if "processed_year_columns" not in st.session_state:
            st.session_state.processed_year_columns = set()

        # Upload-section
        st.markdown('<div class="data-section">Upload your files here.</div>',unsafe_allow_html=True)
        uploaded_file = st.file_uploader(" ", type=["csv", "xlsx"]) # Upload dataset
        
        if uploaded_file is not None:
            if st.session_state.uploaded_file_name != uploaded_file.name:
                df, error = LoadData(uploaded_file)

                if error:
                    st.error(f"Error: {error}") # Print error message if occurs
                else:
                    st.session_state.df = df
                    st.session_state.uploaded_file_name = uploaded_file.name
            
        df = st.session_state.df # Use df from session_statestate

        if df is not None:        
            # Raw data-section
            st.markdown('<div> <h3 class="subheader">DATA</h3></div>', unsafe_allow_html=True)
            st.markdown('<div class="data-preview-section">Here is your data preview.</div>',unsafe_allow_html=True)
            st.write(df.head(10)) # Print the first 10 rows


            # Ask about extracting numbers
            st.markdown('<div class="question-1">Are there any columns where you want to extract numeric values from text?</div>', unsafe_allow_html=True)
            extract_numbers = st.radio(" ", ["Yes", "No"], key="extract_numbers")

            # Ask about datetime
            st.markdown('<div class="question-1">Are there any columns where you want to extract the year from a datetime format?</div>', unsafe_allow_html=True)
            extract_datetime = st.radio("  ", ["Yes", "No"], key="datetime")

            # Select columns
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            selected_columns = [] # Store selected columns

            if extract_numbers == "Yes":
                st.markdown('<div class="extract">Select columns to extract numbers</div>', unsafe_allow_html=True)
                selected_columns = st.multiselect(" ", text_columns, key="number")
           
            # Select columns
            date_columns = df.select_dtypes(include=['object', 'datetime']).columns.tolist()
            selected_year_columns = []  # Store selected columns

            
            if extract_datetime == "Yes":
                st.markdown('<div class="extract">Select columns to extract year</div>', unsafe_allow_html=True)
                selected_year_columns = st.multiselect(" ", date_columns, key="year")

            if st.button("Process data"):
                if extract_numbers == "Yes" and selected_columns:
                    df = CorrectType.extract_numbers_from_text(df, selected_columns)
                
                if extract_datetime == "Yes" and selected_year_columns:
                    df = CorrectType.extract_year(df, selected_year_columns)
                # Always correct dtypedtype
                df = CorrectType.correct_dtype(df, df.columns)
                st.write(df)

            # Explore variabless
            st.markdown('<div class="explore-section">Select to explore.</div>',unsafe_allow_html=True)
            plot_type = st.radio(" ", ["Single variable distribution", "Relationship between 2 variables"])
            
            if plot_type == "Single variable distribution":
                exclude_cols = re.compile(r'(id|ID|date|DATE|time|TIME)', re.IGNORECASE)
                filtered_cols = [col for col in df.columns if not exclude_cols.search(col)]

                if not filtered_cols:
                    st.warning("⚠️ No valid columns available for visualization after filtering out ID, date, and time columns.")
                else:
                    filtered_df = df[filtered_cols]

                    if st.button("Generate chart"):
                        figs = Visualisation.single_value_distribution(filtered_df)
                        if isinstance(figs, list):
                            for fig in figs:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.plotly_chart(figs, use_container_width=True)
            
            elif plot_type == "Relationship between 2 variables":
                st.markdown('<div class="plot-instruction">Select two variables</div>', unsafe_allow_html=True)
                
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                cat_cols = df.select_dtypes(include=['string', 'object']).columns.tolist()
                all_cols = num_cols + cat_cols
                
                selected_col = st.multiselect(" ", all_cols)

                if st.button("Generate chart"):
                    if len(selected_col) < 2:
                        st.warning("⚠️ Please select exactly 2 variables to explore their relationship.")
                    elif len(selected_col) > 2:
                        st.warning("⚠️ Please select only 2 variables.")
                    elif not any(col in num_cols for col in selected_col):
                        st.warning("⚠️ Please select one numerical variable.")
                    else: 
                        numerical_var = next(col for col in selected_col if col in num_cols)
                        second_var = next((col for col in selected_col if col != numerical_var), None)

                        # Check if either column contains "year" or "month" in the name
                        def contains_time_keywords(col):
                            return bool(re.search(r'(year|month)', col, re.IGNORECASE))

                        if any(contains_time_keywords(col) for col in selected_col):
                            fig = Visualisation.bar_count(df, selected_col[0], selected_col[1])
                        else:
                            fig = Visualisation.boxplot(df, selected_col[0], selected_col[1])


                        st.plotly_chart(fig)


            # Clean data
            st.markdown('<div class="clean-section">Select to clean.</div>',unsafe_allow_html=True)


            # Options for cleaning
            ## Lowercase
            lowercase = st.checkbox("Convert text to lowercase")
            lowercase_columns = []
            if lowercase:
                st.markdown('<div class="cleaning-option">Select columns to convert to lowercase</div>', unsafe_allow_html=True)
                lowercase_columns = st.multiselect(" ", df.select_dtypes(include=['object']).columns, key="lowercase")

            ## Remove spaces
            remove_space = st.checkbox("Trim whitespaces if found")

            ## Handle missing values
            missing_values = st.checkbox("Handle missing values")

            ## Remove duplicates
            duplicate = st.checkbox("Remove duplicate rows")
            
            ## Handle outliers
            outliers = st.checkbox("Handle outliers")
            outliers_columns = []
            if outliers:
                st.markdown('<div class="cleaning-option">Select columns to remove outliers</div>', unsafe_allow_html=True)
                outliers_columns = st.multiselect(" ", df.select_dtypes(include=['number']).columns, key="outliers")


            
            # Clean button 
            if st.button("Clean"):
                tool = Tool()
                message = [] # Store messages for each step

                if lowercase and lowercase_columns:
                    df = tool.convert_to_lowercase(df, lowercase_columns)

                if remove_space:
                    df = tool.remove_spaces(df)

                if missing_values:
                    df, removed = tool.handle_missing_values(df)
                    if removed > 0:
                        message.append(f'Removed {removed} missing value rows.')
                    else: 
                        message.append('No missing values detected.')

                if duplicate:
                    df, removed = tool.remove_duplicates(df)
                    if removed > 0:
                        message.append(f'Removed {removed} duplicate rows.')
                    else: 
                        message.append('No duplicate rows detected.')
                
                if outliers and outliers_columns:
                    df, removed = tool.handle_outliers(df, outliers_columns)
                    if removed > 0:
                        message.append(f'Removed {removed} outlier rows.')
                    else: 
                        message.append('No significant ouliers detected.')

                st.success("Success!")
                for msg in message:
                    st.info(msg)
                
                # Download cleaned data
                cleaned_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="&#x2B73;  Download cleaned data",
                    data=cleaned_data,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )

app = App()
app.run()