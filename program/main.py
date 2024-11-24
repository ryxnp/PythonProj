import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Function to load the dataset
def load_dataset():
    try:
        df = pd.read_csv('processed_dataset.csv')
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

# Load the dataset
df = load_dataset()

class MainScreen:
    def __init__(self, master):
        self.master = master
        self.frame = ttk.Frame(master)
        self.frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

        # Create StringVars for dropdown selections
        self.color_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.sentiment_var = tk.StringVar()

        # Create dropdown menus for filtering
        ttk.Label(self.frame, text="Select Color:").grid(row=0, column=0, padx=5)
        self.color_dropdown = ttk.Combobox(self.frame, textvariable=self.color_var)
        self.color_dropdown['values'] = ['Select All'] + df['Color'].unique().tolist()
        self.color_dropdown.bind('<<ComboboxSelected>>', lambda event: self.filter_data())
        self.color_dropdown.grid(row=0, column=1, padx=5)

        ttk.Label(self.frame, text="Select Model:").grid(row=0, column=2, padx=5)
        self.model_dropdown = ttk.Combobox(self.frame, textvariable=self.model_var)
        self.model_dropdown['values'] = ['Select All'] + df['model'].unique().tolist()
        self.model_dropdown.bind('<<ComboboxSelected>>', lambda event: self.filter_data())
        self.model_dropdown.grid(row=0, column=3, padx=5)

        ttk.Label(self.frame, text="Select Sentiment:").grid(row=0, column=4, padx=5)
        self.sentiment_dropdown = ttk.Combobox(self.frame, textvariable=self.sentiment_var)
        self.sentiment_dropdown['values'] = ['Select All'] + df['Sentiment'].unique().tolist()
        self.sentiment_dropdown.bind('<<ComboboxSelected>>', lambda event: self.filter_data())
        self.sentiment_dropdown.grid(row=0, column=5, padx=5)

        # Create buttons for Analysis and Summary screens
        ttk.Button(self.frame, text="Analysis", command=self.show_analysis_screen).grid(row=0, column=6, padx=(5))
        ttk.Button(self.frame, text="Summary", command=self.show_summary_screen).grid(row=0, column=7, padx=(5))

        # Create Treeview for displaying data
        self.tree_frame = ttk.Frame(master)
        self.tree_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')
        
        self.tree = ttk.Treeview(self.tree_frame, columns=list(df.columns), show='headings')
        for col in df.columns:
            self.tree.heading(col, text=f"{col}")
            self.tree.column(col, width=int(70))

        scrollbar = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=scrollbar.set)

        self.tree.pack(side='top', fill='both', expand=True)
        scrollbar.pack(side='bottom', fill='x')

        # Insert initial data into the Treeview
        self.filter_data()
        
    def filter_data(self):
        filtered_df = df.copy()
        
        color_selected = self.color_var.get()
        model_selected = self.model_var.get()
        sentiment_selected = self.sentiment_var.get()

        if color_selected == "Select All":
            color_selected = None
        if model_selected == "Select All":
            model_selected = None
        if sentiment_selected == "Select All":
            sentiment_selected = None

        if color_selected:
            filtered_df = filtered_df[filtered_df['Color'] == color_selected]
        
        if model_selected:
            filtered_df = filtered_df[filtered_df['model'] == model_selected]
        
        if sentiment_selected:
            filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment_selected]

        for row in self.tree.get_children():
            self.tree.delete(row)
        
        for index, row in filtered_df.iterrows():
            self.tree.insert("", "end", values=row.tolist())

    def show_analysis_screen(self):
        AnalysisScreen(self.master)

    def show_summary_screen(self):
       # Hide all previous frames when summary screen is shown.
       for widget in self.master.winfo_children():
           widget.grid_forget()

       # Create a new frame for summary screen directly in the master window.
       summary_frame = ttk.Frame(self.master)
       summary_frame.grid(row=0,column=0,padx=(10),pady=(5),sticky='ew')

       # Create a label for the title
       ttk.Label(summary_frame,text="Summary Screen", font=("Arial", 16)).pack(pady=(20))

       # Create a frame for two-column layout using grid.
       summary_content_frame = ttk.Frame(summary_frame)  
       summary_content_frame.pack(pady=(10))

       # Column 1: Feature Extraction & Clustering
       feature_label = ttk.Label(summary_content_frame, text="Feature Extraction & Clustering", font=("Arial", 14))
       feature_label.grid(row=0, column=0, padx=(10))

        # Ensure that the necessary column is present
       if 'cleaned_reviews' not in df.columns:
            raise ValueError("The dataset must contain 'cleaned_reviews' column.")

        # Feature extraction using TF-IDF
       vectorizer = TfidfVectorizer(max_df=0.5, min_df=5, stop_words='english')
       X_tfidf = vectorizer.fit_transform(df['cleaned_reviews'])

        # Apply K-Means clustering with explicit n_init parameter
       n_clusters = 5  # Define the number of clusters you want to create
       kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
       kmeans.fit(X_tfidf)

        # Assign cluster labels to each review
       df['cluster'] = kmeans.labels_

        # Create a summary table for clusters with sample reviews
       summary_table = []
       for i in range(n_clusters):
            sample_reviews = df[df['cluster'] == i]['cleaned_reviews'].sample(5, random_state=42).tolist()
            summary_table.append({'Cluster': i, 'Sample Reviews': sample_reviews})

        # Convert list of dictionaries to DataFrame for better display
       summary_table_df = pd.DataFrame(summary_table)

        # Create a Treeview to display cluster summaries
       cluster_treeview = ttk.Treeview(summary_content_frame)

        # Define columns for Treeview (Cluster and Sample Reviews)
       cluster_treeview["columns"] = ("Cluster", "Sample Reviews")
       cluster_treeview.column("#0", width=0)  # Hide first empty column
       cluster_treeview.column("Cluster", anchor=tk.W, width=100)
       cluster_treeview.column("Sample Reviews", anchor=tk.W)

       cluster_treeview.heading("#0", text="", anchor=tk.W)  # Empty header
       cluster_treeview.heading("Cluster", text="Cluster", anchor=tk.W)
       cluster_treeview.heading("Sample Reviews", text="Sample Reviews", anchor=tk.W)

        # Insert results into Treeview
       for entry in summary_table:
            cluster_treeview.insert("", "end", values=(entry['Cluster'], ', '.join(entry['Sample Reviews'])))

       cluster_treeview.grid(row=1, column=0, padx=(10))  # Place Treeview in first column

       # Column 2: Linear Regression Results Display Box (as a Treeview)
       regression_label = ttk.Label(summary_content_frame,text="Linear Regression Results", font=("Arial", 14))
       regression_label.grid(row=(0),column=(1),padx=(10))

       # Create a Treeview to display results from logistic regression analysis.
       regression_treeview = ttk.Treeview(summary_content_frame)
       
       # Define columns for Treeview (Classification Report and Confusion Matrix)
       regression_treeview["columns"] = ("Metric", "Value")
       regression_treeview.column("#0", width=0)  # Hide first empty column
       regression_treeview.column("Metric", anchor=tk.W, width=120)
       regression_treeview.column("Value", anchor=tk.W)

       regression_treeview.heading("#0", text="", anchor=tk.W)  # Empty header
       regression_treeview.heading("Metric", text="Metric", anchor=tk.W)
       regression_treeview.heading("Value", text="Value", anchor=tk.W)

       # Insert results into Treeview (initially empty)
       results_data = perform_logistic_regression()
       
       for metric_name, metric_value in results_data.items():
           regression_treeview.insert("", "end", values=(metric_name, metric_value))

       regression_treeview.grid(row=(1),column=(1),padx=(10))  # Place Treeview in second column

       # Back button to return to main screen
       ttk.Button(summary_frame,text="Back to Analysis",
                  command=lambda: AnalysisScreen(self.master)).pack(pady=(10))
       
        # Visualization of clusters using PCA
       pca = PCA(n_components=2)
       X_reduced = pca.fit_transform(X_tfidf.toarray())

       plt.figure(figsize=(10, 6))
       sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=df['cluster'], palette='viridis', legend='full')
       plt.title('K-Means Clustering of Reviews')
       plt.xlabel('PCA Component 1')
       plt.ylabel('PCA Component 2')
       plt.show()

def perform_logistic_regression():
    """Perform logistic regression and return the results as a dictionary."""
    try:
         # Ensure that necessary columns are present in the DataFrame.
         if 'cleaned_reviews' not in df.columns or 'Sentiment' not in df.columns:
             raise ValueError("The dataset must contain 'cleaned_reviews' and 'Sentiment' columns.")

         # Split the dataset into features and target variable.
         X = df['cleaned_reviews']
         y = df['Sentiment']

         # Stratified split to maintain class distribution.
         X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                             test_size=0.2,
                                                             random_state=42,
                                                             stratify=y)

         # Feature extraction using TF-IDF.
         vectorizer = TfidfVectorizer()
         X_train_tfidf = vectorizer.fit_transform(X_train)
         X_test_tfidf = vectorizer.transform(X_test)

         # Train a logistic regression model with balanced class weights to handle class imbalance.
         model = LogisticRegression(class_weight='balanced', max_iter=1000)  # Increased max_iter for convergence.
         model.fit(X_train_tfidf, y_train)

         # Make predictions on the test set.
         y_pred = model.predict(X_test_tfidf)

         # Evaluate the model's performance and format results.
         report_dict = classification_report(y_test, y_pred, output_dict=True)  # Get report as dictionary.

         confusion_mat = confusion_matrix(y_test, y_pred).flatten()  # Flatten confusion matrix.

         results_dict = {
             "Accuracy": report_dict["accuracy"],
             "Precision": report_dict["weighted avg"]["precision"],
             "Recall": report_dict["weighted avg"]["recall"],
             "F1 Score": report_dict["weighted avg"]["f1-score"],
             "True Positives": confusion_mat[1],
             "False Positives": confusion_mat[2],
             "True Negatives": confusion_mat[0],
             "False Negatives": confusion_mat[3]
         }

         return results_dict

    except Exception as e:
          return {"Error": str(e)}

class AnalysisScreen:
    def __init__(self, master):
       # Hide all previous frames when analysis screen is shown.
       for widget in master.winfo_children():
           widget.grid_forget()

       # Create a new frame for analysis screen
       frame = ttk.Frame(master)
       frame.grid(row=0,column=0,padx=(10),pady=(5),sticky='ew')

       # Analysis controls in analysis_frame - single row layout with buttons and dropdowns
       ttk.Label(frame,text="Select Analysis Type:").grid(row=0,column=1,padx=(10))
       
       analysis_var = tk.StringVar()
       analysis_dropdown = ttk.Combobox(frame,textvariable=analysis_var)

       analysis_options = {
           "Scatter Plot": ("ratingScore", "Sentiment", "date"),
           "Box Plot": ("Color","Storage", "Sentiment", "ratingScore"),
           "Pie Chart": ("ratingScore","Sentiment", "model", "Color", "Storage"),
           "Violin Plot": ("Color", "Storage","ratingScore"),
           "Pair Plot": ("productAsin", "date", "isVerified", "ratingScore", 
                         "reviewTitle", "reviewDescription", 
                         "model", "Color", "Storage", 
                         "Sentiment", "cleaned_reviews")
       }

       analysis_dropdown['values'] = list(analysis_options.keys())
       analysis_dropdown.grid(row=0,column=2,padx=(10))

       x_var = tk.StringVar()
       x_dropdown = ttk.Combobox(frame,textvariable=x_var) 
       x_dropdown.grid(row=0,column=4,padx=(10))

       y_var = tk.StringVar() 
       y_dropdown = ttk.Combobox(frame,textvariable=y_var) 
       y_dropdown.grid(row=0,column=6,padx=(10))

       # Bind selection event to dynamically update X and Y options based on selected analysis type.
       analysis_dropdown.bind('<<ComboboxSelected>>', lambda event: update_xy_options(event, analysis_options, x_dropdown, y_dropdown))

       ttk.Button(frame,text="Perform Analysis",
                   command=lambda: perform_analysis(analysis_var.get(), x_var.get(), y_var.get())).grid(row=0,column=7,padx=(1),pady=(5))
       
       ttk.Button(frame,text="Back to Menu",
                   command=lambda: MainScreen(master)).grid(row=0,column=0,pady=(5))


def update_xy_options(event, analysis_options, x_dropdown, y_dropdown):
    """Update X and Y dropdown options based on selected analysis type."""
    selected_analysis_type = event.widget.get()
    x_options = []
    y_options = []

    if selected_analysis_type in analysis_options:
        valid_columns = analysis_options[selected_analysis_type]
        
        x_options.append("Select X Value")  # Placeholder option
        y_options.append("Select Y Value")  # Placeholder option
        
        x_options += valid_columns
        
        if len(valid_columns) > 1:
            y_options += valid_columns[1:]  # Exclude the first option from Y options

    x_dropdown['values'] = x_options
    x_dropdown.current(0)  # Set default selection
    
    y_dropdown['values'] = y_options
    y_dropdown.current(0)  # Set default selection

def perform_analysis(analysis_type,x_value,y_value):
    if analysis_type == "Pie Chart":
         if not x_value or x_value == "Select X Value":
             messagebox.showwarning("Input Error","Please select an X value for the Pie Chart.")
             return

         df[x_value].value_counts().plot.pie(autopct='%1.1f%%')
         plt.title(f'Pie Chart of {x_value}')
         plt.ylabel('')
         plt.show()
         return

    if not x_value or not y_value or x_value == "Select X Value" or y_value == "Select Y Value":
         messagebox.showwarning("Input Error","Please select both X and Y values.")
         return

    if analysis_type == "Scatter Plot":
         plt.scatter(df[x_value], df[y_value])
         plt.title(f'Scatter Plot of {y_value} vs {x_value}')
         plt.xlabel(x_value)
         plt.ylabel(y_value)
         plt.show()

    elif analysis_type == "Box Plot":
         sns.boxplot(x=x_value,y=y_value,data=df)
         plt.title(f'Box Plot of {y_value} by {x_value}')
         plt.show()

    elif analysis_type == "Violin Plot":
         sns.violinplot(x=x_value,y=y_value,data=df)
         plt.title(f'Violin Plot of {y_value} by {x_value}')
         plt.show()

    elif analysis_type == "Pair Plot":
         sns.pairplot(df)
         plt.title('Pair Plot of DataFrame')
         plt.show()

# Start the Tkinter event loop
root = tk.Tk()
root.title("Basic GUI Layout")
root.maxsize(1000, 1000)
root.config(bg="grey")

# Initialize the main screen instance to start with it.
MainScreen(root)

root.mainloop()