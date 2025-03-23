import pandas as pd
import streamlit as st
import re
from fuzzywuzzy import process
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

class CorrectType:
  @staticmethod
  def extract_numbers_from_text(df, columns):
    """
    Extracts numeric values from a text-based Pandas column.
    Handles integers, decimals, negative values, and comma formatting.
    """
    for col in columns:
      if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '', regex=True)  # Remove commas
        number = df[col].str.extract(r'([-+]?\d*\.\d+|\d+)')[0]  # Extract first number found
        df[col] = pd.to_numeric(number, errors='coerce')
    return df 
  

  @staticmethod
  def extract_year(df, selected_year_columns):
    """
    Extracts the year from specified datetime columns and store it in a new column {col}_year
    Uses `st.session_state` to track processed columns
    Prevents duplicate extraction (does not overwrite existing {col}_year)
    """
    # Initialize session state if not exists
    if "processed_columns" not in st.session_state:
      st.session_state.processed_columns = set()

    for col in selected_year_columns:
      year_col = f"{col}_year"

      if year_col in st.session_state.processed_columns:
        continue  # Skip if already processed

      if col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
          df[col] = pd.to_datetime(df[col], errors='coerce')

        df[year_col] = df[col].dt.year.dropna().astype(int)  # Ensure int type
        st.session_state.processed_columns.add(year_col)

    return df


  @staticmethod
  def correct_dtype(df, columns):
    """
    Automatically detects and corrects column data types in a DataFrame:
    - Converts numeric values stored as objects to appropriate numeric types 
    (If users do not select "Extract numbers," numeric columns stored as text will still be corrected)
    - Ensures columns containing any non-numeric characters remain as strings.
    """
    for col in columns:
        original_dtype = df[col].dtype  # Original dtype
        
        # Try converting to numeric first
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        if numeric_col.notna().sum() > 0:  # Only convert if values are valid
            df[col] = numeric_col
            continue 

        # If not numeric, astype string
        if original_dtype == 'object':
          df[col] = df[col].astype(str)
        else:
          df[col] = df[col]

    return df
  


class Visualisation:
  @staticmethod
  def single_value_distribution(df):
    """
    Automatically detects the data type of the selected column and plots 
    appropriate graphs (histogram for numerical, bar/pie chart for categorical).
    """

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['string', 'object']).columns.tolist()

    # Define palette
    palette = ['#565695', "#EC98BE", "#779ecb", "#b36389", "#8b88cc", "#e56f92", "#98bad5", "#63859e"]
    color = ["#EC98BE"]

    figs = []
    plot_types = [] 

    # Categorical data: Bar/pie chart
    for col in cat_cols:
      value_counts = df[col].value_counts().sort_values(ascending=False)

      if len(value_counts) >= 15:
        continue
      
      # Pie chart
      if len(value_counts) <= 4:
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Distribution of {col}",
            color=value_counts.index.astype(str),
            color_discrete_sequence=['#D98324', "#EFDCAB", "#443627", "#F2F6D0"]
        )
        fig.update_layout(width=900,
                          height=900,
                          margin=dict(t=50, b=50, l=50, r=50))
        plot_types.append("domain") 
      else:
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Distribution of {col}",
            color=value_counts.index,
            color_discrete_sequence=palette
        )

        # Update layout to add bottom margin
        fig.update_layout(
            margin=dict(t=100, b=100, l=50, r=50),
            showlegend=True
        )
        plot_types.append("xy")

      # Append to figs
      figs.append(fig)

    # Numerical data: Histogram or bar chart
    for col in [c for c in num_cols if "year" not in c.lower()]: # Skip columns with "year" since they are already plotted as a line graph 
      unique_values = df[col].nunique()
      
      if unique_values <= 10:
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = ['Category', 'Count']
        value_counts["Category"] = value_counts["Category"].astype(int)
        value_counts["Category"] = value_counts["Category"].astype(str)
        
        # Bar chart
        palette = ['#565695', "#EC98BE", "#779ecb", "#b36389", "#8b88cc", "#e56f92", "#98bad5", "#63859e"]
        fig = px.bar(
                value_counts,
                x="Category",
                y="Count",
                color="Category", 
                color_discrete_sequence=palette,  
                title=f"Distribution of {col}"
        )
        fig.update_traces(
            text=value_counts.values,
            textposition='outside')
        plot_types.append("xy")
      else:
        fig = px.histogram(
            df, 
            x=col, 
            title=f"Distribution of {col}",
            nbins=30,
            color_discrete_sequence=color
        )
        plot_types.append("xy")
      figs.append(fig)
    
    if len(figs) == 1:
      return figs[0]
    else:
      rows = (len(figs) // 2) + (len(figs) % 2)

      specs = []
      for i in range(rows):
        row_spec = []
        for j in range(2):
          index = i * 2 + j
          if index < len(figs):
            row_spec.append({"type": plot_types[index]})
          else:
            row_spec.append(None)  # Empty cell
        specs.append(row_spec)
      fig_subplots = make_subplots(rows=rows,
                                  cols=2,
                                  subplot_titles=[f.layout.title.text for f in figs],
                                  specs=specs,
                                  vertical_spacing=0.07)

      for i, f in enumerate(figs):
        for trace in f.data:
          fig_subplots.add_trace(trace, row=(i // 2) + 1, col=(i % 2) + 1)

      fig_subplots.update_layout(showlegend=False,
                                height=250 * rows,
                                width=900,
                                margin=dict(t=50, b=50, l=50, r=50))
      return fig_subplots


  @staticmethod
  def boxplot(df, var1, var2):
    """
    Create a boxplot, ensuring that the numerical variable is always on the y-axis.
      var1: First selected variable
      var2: Second selected variable
    """
    # Drop missing values
    df = df[[var1, var2]].dropna()

    # Define which variable is numeric
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if var1 in num_cols and var2 not in num_cols:
      second_var, numerical_var  = var1, var2
    elif var2 in num_cols and var1 not in num_cols:
      second_var, numerical_var = var1, var2
    elif var1 in num_cols and var2 in num_cols:
      second_var, numerical_var = var1, var2
    else:
      raise ValueError("At lease one variable must be numerical.")

    # Boxplot
    fig = px.box(df, 
                x=numerical_var, 
                y=second_var, 
                title=f"Distribution of {second_var} by {numerical_var}",
                color=numerical_var,
                template="plotly_white")
    fig.update_layout(xaxis_title=numerical_var,
                  yaxis_title=second_var,
                  showlegend=False,
                  title_x=0.36)
    return fig
  
  @staticmethod
  def bar_count(df, var1, var2):
    """
    Create a bar chart if one of the variables contains "year" or "month".
    """
    # Drop missing values
    df = df[[var1, var2]].dropna()

    # Define numerical columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Ensure categorical variable selection
    if var1 in num_cols and var2 not in num_cols:
        numerical_var, second_var = var1, var2
    elif var2 in num_cols and var1 not in num_cols:
        numerical_var, second_var = var2, var1
    elif var1 in num_cols and var2 in num_cols:
        
        # Force the column with "year" or "month" to be categorical if both are numerical
        if re.search(r'\b(year|month)\b', var1, re.IGNORECASE):
            second_var, numerical_var = var1, var2
        else:
            second_var, numerical_var = var2, var1
    else:
        raise ValueError("At least one variable must be numerical.")

    value_counts = df[second_var].value_counts().sort_index()
    palette = ['#565695', "#EC98BE", "#779ecb", "#b36389", "#8b88cc", "#e56f92", "#98bad5", "#63859e"]
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        color=value_counts.index.astype(str),
        color_discrete_sequence=palette,  
        title=f"Number of {numerical_var} over time",
        labels={"x": second_var, "y": "Count"}
    )
    fig.update_layout(title_x=0.4,
                      showlegend=False)
    fig.update_traces(text=value_counts.values, textposition='outside')

    return fig



class Tool:
  def __init__(self, missing_strategy='auto', contamination=0.05):
    self.missing_strategy = missing_strategy
    self.imputer = None

  def convert_to_lowercase(self, df):
    """
      Convert users' selected columns to lowercase.
    """
    text_columns = df.select_dtypes(include=['object']).columns
    valid_columns = [col for col in selected_columns if col in text_columns] 

    for col in valid_columns:
      df[col] = df[col].str.lower()
    return df


  def remove_spaces(self, df):
    """
    Remove leading and trailing spaces.
    """
    df.columns = df.columns.str.strip()
    return df
  

  def fix_inconsistency(self, df, threshold = 85):
    """
    Fix text inconsistencies using fuzzy matching.
    """
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
      unique_values = df[col].dropna().unique()
      corrected_values = {}

      for val in unique_values:
        best_match = process.extractOne(val, unique_values)
        if best_match and best_match[1] >= threshold:  # Match confidence >= threshold
          corrected_values[val] = best_match[0]

      df[col] = df[col].replace(corrected_values) 
    return df


  def remove_duplicates(self, df):
    """
    Remove duplicate rows with count of removed records.
    """
    rows_before = df.shape[0] # Count rows before cleaning
    df = df.drop_duplicates() 
    rows_after = df.shape[0] # Count rows after cleaning
    records_removed = rows_before - rows_after

    return df, records_removed


  def handle_missing_values(self, df):
    """
    - Automatically fills missing values based on data type.
    - Count the number of removed records.
    """
    
    rows_before = df.shape[0] # Count rows before cleaning

    # Separate into numerical and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Remove columns with > 50% missing values
    threshold = 0.5
    missing_ratio = df.isnull().sum() / len(df)
    df = df.drop(columns=missing_ratio[missing_ratio > threshold].index)

    # Dictionary to store imputers for numerical columns
    self.imputer = {}

    # Numerical values: Fill missing values with mean/median based on distribution
    for col in numeric_cols:
      if df[col].skew() > 1: # If highly skewed, use median
        self.imputer[col] = SimpleImputer(strategy='median')
      else: # If numerical values are normally distributed, use mean
        self.imputer[col] = SimpleImputer(strategy='mean')
      
      # Replace missing values in the column with the computed median or mean
      df[col] = self.imputer[col].fit_transform(df[[col]])
    
    # Categorical values: Fill missing values with "Unknown"
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    rows_after = df.shape[0] # Count rows after cleaning
    records_removed = rows_before - rows_after

    return df, records_removed
  
  
  def handle_outliers(self, df, columns):
    """
    Remove outliers using Isolation Forest with contamination = 0.05
    """

    rows_before = df.shape[0] # Count rows before cleaning

    # Initialize and fit Isolation Forest model
    clf = IsolationForest(contamination=0.05)
    df['anomaly'] = clf.fit_predict(df.select_dtypes(include=['number']))

    # Keep only normal data (anomaly == 1) and drop anomalies (outliers)
    df = df[df['anomaly'] == 1].drop(columns=['anomaly'])

    rows_after = df.shape[0] # Count rows after cleaning
    records_removed = rows_before - rows_after
    
    return df, records_removed

