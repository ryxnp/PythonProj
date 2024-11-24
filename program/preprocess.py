import pandas as pd
import re
from textblob import TextBlob
import nltk
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_dataset():
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv('iphone_reviews.csv')
        print("Loaded dataset columns:", df.columns.tolist())  # Debugging statement
        return df  # Return the DataFrame itself for processing
    except Exception as e:
        messagebox.showerror("Error", f"Error loading dataset: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def analyze_sentiment(review):
    """Analyze sentiment of a review."""
    if isinstance(review, str):
        analysis = TextBlob(review)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    else:
        return 'Unknown'

def preprocess_text(text):
    """Preprocess review text by cleaning and lemmatizing."""
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
        text = text.lower()  # Convert to lowercase
        tokens = text.split()  # Tokenization
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization
        return ' '.join(tokens)  # Join tokens back into a string

def process_reviews():
    """Process reviews by analyzing sentiment and cleaning text."""
    global df  # Declare df as global to access it in save_to_csv function
    df = load_dataset()

    if not df.empty:
        # Check if 'reviewDescription' exists in DataFrame
        if 'reviewDescription' not in df.columns:
            messagebox.showerror("Error", "Column 'reviewDescription' not found in dataset.")
            return
        
        # Adding new columns for sentiment and cleaned reviews directly to the original DataFrame
        df['Sentiment'] = df['reviewDescription'].apply(analyze_sentiment)
        df['cleaned_reviews'] = df['reviewDescription'].apply(preprocess_text)

        # Debugging statement to check new columns creation
        print("Columns after processing:", df.columns.tolist())
        
        # Clear existing data in the treeview
        for row in tree.get_children():
            tree.delete(row)

        # Insert new data into the treeview from updated DataFrame
        for index, row in df.iterrows():
            tree.insert("", "end", values=(row['reviewDescription'], row['cleaned_reviews'], row['Sentiment']))

        # Count positive and negative reviews
        positive_count = len(df[df['Sentiment'] == 'Positive'])
        negative_count = len(df[df['Sentiment'] == 'Negative'])

        # Insert summary row
        tree.insert("", "end", values=("Total Positive Reviews", positive_count, ""))
        tree.insert("", "end", values=("Total Negative Reviews", negative_count, ""))

    else:
        messagebox.showwarning("Warning", "No data available to process.")

def save_to_csv():
    """Save the updated DataFrame to a CSV file with only Positive and Negative sentiments."""
    global df  # Use the updated global DataFrame with new columns
    
    if not df.empty:
        try:
            # Ensure that the new columns exist before saving
            required_columns = ['reviewDescription', 'cleaned_reviews', 'Sentiment']
            for col in required_columns:
                if col not in df.columns:
                    messagebox.showerror("Error", f"Column '{col}' not found in DataFrame.")
                    return
            
            # Filter the DataFrame to keep only Positive and Negative sentiments
            filtered_df = df[df['Sentiment'].isin(['Positive', 'Negative'])]

            # Check if the filtered DataFrame is empty
            if filtered_df.empty:
                messagebox.showwarning("Warning", "No Positive or Negative reviews to save.")
                return
            
            # Save the filtered DataFrame containing only relevant rows
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                       filetypes=[("CSV Files", "*.csv")])
            if file_path:
                filtered_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", "Data saved successfully.")
                print("Saved DataFrame columns:", filtered_df.columns.tolist())  # Debugging statement
        except Exception as e:
            messagebox.showerror("Error", f"Error saving file: {e}")

# Create the main window using Tkinter
window = tk.Tk()
window.title("iPhone Reviews Dataset Pre-Processing")
window.geometry("800x600")

# Create buttons for processing and saving reviews
process_button = tk.Button(window, text="Process Dataset", command=process_reviews)
process_button.pack(pady=10)

save_button = tk.Button(window, text="Download CSV", command=save_to_csv)
save_button.pack(pady=5)

# Create a Treeview to display reviews
tree = ttk.Treeview(window, columns=("Original Review", "Cleaned Review", "Sentiment"), show='headings')
tree.heading("Original Review", text="Original Review")
tree.heading("Cleaned Review", text="Cleaned Review")
tree.heading("Sentiment", text="Sentiment")
tree.pack(expand=True, fill='both', padx=10, pady=10)

# Start the Tkinter event loop
window.mainloop()