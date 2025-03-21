import pandas as pd

def LoadData(uploaded_file):
  try:
    df = pd.read_csv(uploaded_file)
    return df, None
  except Exception as e:
    raise ValueError(f"Error loading file: {e}")