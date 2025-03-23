import pandas as pd
import io

def LoadData(uploaded_file):
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Kiểm tra loại file
        if file_extension == 'csv':
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')  # Try with another encoding
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        else:
            return None, "❌ Unsupported file type. Please upload a CSV or Excel file."

        # Check empty filefile
        if df.empty:
            return None, "⚠️ The uploaded file is empty. Please check your data."

        return df, None

    except Exception as e:
        return None, f"❌ Error loading file: {str(e)}"
