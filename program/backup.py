import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt
import seaborn as sns  # For additional plots like violin plots and pair plots

# Function to load the dataset
def load_dataset():
    try:
        # Load CSV file into a DataFrame
        df = pd.read_csv('processed_dataset.csv')
        return df  # Return the DataFrame itself for processing
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Load the dataset
df = load_dataset()

# Function to filter the DataFrame based on selected dropdown values
def filter_data():
    filtered_df = df.copy()
    
    # Get selected values from dropdowns
    color_selected = color_var.get()
    model_selected = model_var.get()
    sentiment_selected = sentiment_var.get()
    
    # Apply filters based on selected values
    if color_selected == "Select All":
        color_selected = None  # Reset to None for filtering all colors
    if model_selected == "Select All":
        model_selected = None  # Reset to None for filtering all models
    if sentiment_selected == "Select All":
        sentiment_selected = None  # Reset to None for filtering all sentiments

    if color_selected:
        filtered_df = filtered_df[filtered_df['Color'] == color_selected]
    if model_selected:
        filtered_df = filtered_df[filtered_df['model'] == model_selected]
    if sentiment_selected:
        filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment_selected]

    # Clear the treeview and insert filtered data
    for row in tree.get_children():
        tree.delete(row)
    
    for index, row in filtered_df.iterrows():
        tree.insert("", "end", values=row.tolist())

# Function to handle item selection in the Treeview
def on_tree_select(event):
    selected_item = tree.selection()  # Get selected item
    if selected_item:  # Check if there is a selection
        item_values = tree.item(selected_item)['values']  # Get values of the selected item
        show_custom_alert(item_values)  # Show custom alert box with item details

# Function to show custom alert box with formatted text
def show_custom_alert(item_values):
    alert_window = tk.Toplevel(root)
    alert_window.title("Item Details")
    
    # Create a Text widget for formatted output
    text_widget = tk.Text(alert_window, width=50, height=10)
    text_widget.pack(padx=10, pady=10)

    # Format the output (example: displaying each item in a new line)
    formatted_text = "\n".join([f"{col}: {val}" for col, val in zip(df.columns, item_values)])
    
    text_widget.insert(tk.END, formatted_text)
    
    # Create a Close button to close the alert window
    close_button = tk.Button(alert_window, text="Close", command=alert_window.destroy)
    close_button.pack(pady=(0, 10))

# Function to switch to the analysis screen
def show_analysis_screen():
    main_frame.grid_forget()
    analysis_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')  # Show analysis frame

# Function to go back to the main screen
def show_main_screen():
    main_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')
    column_frame.grid_forget()
    summary_frame.grid_forget()
    analysis_frame.grid_forget()  # Hide analysis frame
    
def show_summary_screen():
    main_frame.grid_forget()  # Hide the main frame
    summary_frame = Frame(root)  # Create a new frame for the summary
    summary_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')  # Place it in the grid

    # Create a label for the title
    Label(summary_frame, text="Summary Screen", font=("Arial", 16)).pack(pady=20)

    # Create a frame for the two-column layout
    column_frame = Frame(summary_frame)
    column_frame.pack(pady=10)

    # Create the first column for Feature Extraction & Clustering
    feature_label = Label(column_frame, text="Feature Extraction & Clustering", font=("Arial", 14))
    feature_label.grid(row=0, column=0, padx=10)

    feature_text = Text(column_frame, width=40, height=10)  # Adjust size as needed
    feature_text.grid(row=1, column=0, padx=10)
    feature_text.insert(END, "Details about Feature Extraction & Clustering...")  # Add relevant content here

    # Create the second column for Linear Regression
    regression_label = Label(column_frame, text="Linear Regression", font=("Arial", 14))
    regression_label.grid(row=0, column=1, padx=10)

    regression_text = Text(column_frame, width=40, height=10)  # Adjust size as needed
    regression_text.grid(row=1, column=1, padx=10)
    regression_text.insert(END, "Details about Linear Regression...")  # Add relevant content here

    # Optionally add a button to go back to the main screen
    Button(summary_frame, text="Back", command=show_main_screen).pack(pady=(10))

# Make sure to call this function when you want to display the summary screen.
    

# Function to display selected analysis graphically
def perform_analysis():
    analysis_type = analysis_var.get()
    x_value = x_var.get()
    y_value = y_var.get()

    # Handle Pie Chart case
    if analysis_type == "Pie Chart":
        # Ensure x_value is selected
        if not x_value or x_value == "Select X Value":
            messagebox.showwarning("Input Error", "Please select an X value for the Pie Chart.")
            return
        
        # Create pie chart based on the selected x_value
        df[x_value].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie Chart of {x_value}')
        plt.ylabel('')  # Hide the y-label for better aesthetics
        plt.show()
        return  # Exit after displaying pie chart

    # For other analysis types, ensure both x and y values are selected
    if not x_value or not y_value or x_value == "Select X Value" or y_value == "Select Y Value":
        messagebox.showwarning("Input Error", "Please select both X and Y values.")
        return

    if analysis_type == "Scatter Plot":
        plt.scatter(df[x_value], df[y_value])
        plt.title(f'Scatter Plot of {y_value} vs {x_value}')
        plt.xlabel(x_value)
        plt.ylabel(y_value)
        plt.show()
    
    elif analysis_type == "Box Plot":
        sns.boxplot(x=x_value, y=y_value, data=df)
        plt.title(f'Box Plot of {y_value} by {x_value}')
        plt.show()
    
    elif analysis_type == "Violin Plot":
        sns.violinplot(x=x_value, y=y_value, data=df)
        plt.title(f'Violin Plot of {y_value} by {x_value}')
        plt.show()
    
    elif analysis_type == "Pair Plot":
        sns.pairplot(df)
        plt.title('Pair Plot of DataFrame')
        plt.show()

# Function to update X and Y dropdown options based on selected analysis type.
def update_xy_options(event):
   """Update X and Y dropdown options based on selected analysis type."""
   selected_analysis_type = analysis_var.get()
   x_options = []
   y_options = []

   if selected_analysis_type in analysis_options:
       x_options.append("Select X Value")  # Placeholder option 
       y_options.append("Select Y Value")  # Placeholder option 

       valid_columns = analysis_options[selected_analysis_type]

       if len(valid_columns) > 0:
           x_options += valid_columns 
           x_dropdown['values'] = x_options  
           x_dropdown.current(0)  

           if len(valid_columns) > 1:
               y_options += valid_columns 
               y_options.remove(valid_columns[0])  
           y_dropdown['values'] = y_options  
           y_dropdown.current(0)  

       else:
           x_dropdown['values']=[]
           y_dropdown['values']=[]

   else:
       x_dropdown['values']=[]
       y_dropdown['values']=[]

# Create the main window
root = tk.Tk()
root.title("Basic GUI Layout")
root.maxsize(1000, 1000)
root.config(bg="grey")

# Create top frame for controls (main screen)
main_frame = Frame(root, bg='white')
main_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

column_frame = Frame(root)
summary_frame = Frame(root)                   


# Create bottom frame for Treeview (main screen)
bottom_frame = Frame(root, bg='white')
bottom_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')

# Create StringVars for dropdown selections for the single row (main screen)
color_var = tk.StringVar()
model_var = tk.StringVar()
sentiment_var = tk.StringVar()

# Create dropdown menu for Color filtering (single row)
Label(main_frame, text="Select Color:").grid(row=0, column=0, padx=5)
color_dropdown = ttk.Combobox(main_frame, textvariable=color_var)
color_dropdown['values'] = ['Select All'] + df['Color'].unique().tolist()  # Add Select All option
color_dropdown.bind('<<ComboboxSelected>>', lambda event: filter_data())
color_dropdown.grid(row=0, column=1, padx=5)

# Create dropdown menu for Model filtering (single row)
Label(main_frame, text="Select Model:").grid(row=0, column=2, padx=5)
model_dropdown = ttk.Combobox(main_frame, textvariable=model_var)
model_dropdown['values'] = ['Select All'] + df['model'].unique().tolist()  # Add Select All option
model_dropdown.bind('<<ComboboxSelected>>', lambda event: filter_data())
model_dropdown.grid(row=0, column=3, padx=5)

# Create dropdown menu for Sentiment filtering (single row)
Label(main_frame, text="Select Sentiment:").grid(row=0, column=4, padx=5)
sentiment_dropdown = ttk.Combobox(main_frame, textvariable=sentiment_var)
sentiment_dropdown['values'] = ['Select All'] + df['Sentiment'].unique().tolist()  # Add Select All option
sentiment_dropdown.bind('<<ComboboxSelected>>', lambda event: filter_data())
sentiment_dropdown.grid(row=0, column=5, padx=5)

# Create a single button named "Analysis"
Button(main_frame, text="Analysis", command=show_analysis_screen).grid(row=0, column=6, padx=5)

Button(main_frame, text="Summary", command=show_summary_screen).grid(row=0, column=7, padx=5)

# Centering: Configure grid weights for centering effect in the single row (main screen)
for i in range(7):
    main_frame.grid_columnconfigure(i, weight=1)  # Allow all columns to expand equally

# Create a Treeview to display the DataFrame in bottom_frame (main screen)
tree = ttk.Treeview(bottom_frame, columns=list(df.columns), show='headings')
for col in df.columns:
    tree.heading(col, text=col)
    tree.column(col, width=70)  # Set a default width for each column

# Add a scrollbar for horizontal scrolling (main screen)
scrollbar = Scrollbar(bottom_frame, orient="horizontal", command=tree.xview)
tree.configure(xscrollcommand=scrollbar.set)

# Pack the Treeview and scrollbar into bottom_frame (main screen)
tree.pack(side='top', fill='both', expand=True)
scrollbar.pack(side='bottom', fill='x')

# Insert initial data into the Treeview (main screen)
filter_data()  # Show all data initially

# Bind selection event to Treeview (main screen)
tree.bind("<<TreeviewSelect>>", on_tree_select)

# Configure grid weights to make frames expand properly (main screen)
root.grid_rowconfigure(0, weight=1)  # Allow top frame to expand vertically
root.grid_rowconfigure(1, weight=3)  # Allow bottom frame to take more space

# Create Analysis Frame (hidden initially)
analysis_frame = Frame(root)

# Analysis controls in analysis_frame - single row layout with buttons and dropdowns
Label(analysis_frame,text="Select Analysis Type:").grid(row=0,column=1,padx=(10)) 
analysis_var = tk.StringVar()
analysis_dropdown = ttk.Combobox(analysis_frame,textvariable=analysis_var)

# Define available analyses and their valid X/Y options.
analysis_options = {
   "Scatter Plot": ("ratingScore", "Sentiment", "date"),
   "Box Plot": ("Color","Storage", "Sentiment", "ratingScore"),
   "Pie Chart": ("ratingScore","Sentiment", "model", "Color", "Storage",),
   "Violin Plot": ("Color", "Storage","ratingScore"),
   "Pair Plot": ("productAsin",	"date",	"isVerified",	
                 "ratingScore",	"reviewTitle",	"reviewDescription",	
                 "model",	"Color",	"Storage",	"Sentiment",	
                 "cleaned_reviews",)
}

analysis_dropdown['values'] = list(analysis_options.keys())  # Add more analysis types as needed
analysis_dropdown.bind('<<ComboboxSelected>>', update_xy_options)  # Bind selection event
analysis_dropdown.grid(row=0,column=2,padx=(10))

Label(analysis_frame,text="Select X Value:").grid(row=0,column=3,padx=(10))
x_var = tk.StringVar()
x_dropdown = ttk.Combobox(analysis_frame,textvariable=x_var) 
x_dropdown.grid(row=0,column=4,padx=(10))

Label(analysis_frame,text="Select Y Value:").grid(row=0,column=5,padx=(10))
y_var = tk.StringVar()
y_dropdown = ttk.Combobox(analysis_frame,textvariable=y_var) 
y_dropdown.grid(row=0,column=6,padx=(10))

Button(analysis_frame,text="Perform Analysis",command=lambda: perform_analysis()).grid(row=0,column=7,padx=(1),pady=(5))

Button(analysis_frame,text="Back",command=lambda: show_main_screen()).grid(row=0,column=0,pady=(5))

def update_xy_options(event):
   """Update X and Y dropdown options based on selected analysis type."""
   selected_analysis_type = analysis_var.get()
   x_options = []
   y_options = []

   if selected_analysis_type in analysis_options:
       x_options.append("Select X Value")  # Placeholder option 
       y_options.append("Select Y Value")  # Placeholder option 

       valid_columns = analysis_options[selected_analysis_type]

       if len(valid_columns) > 0:
           x_options += valid_columns 
           x_dropdown['values'] = x_options  
           x_dropdown.current(0)  

           if len(valid_columns) > 1:
               y_options += valid_columns 
               y_options.remove(valid_columns[0])  
           y_dropdown['values'] = y_options  
           y_dropdown.current(0)  

       else:
           x_dropdown['values']=[]
           y_dropdown['values']=[]

   else:
       x_dropdown['values']=[]
       y_dropdown['values']=[]

# Start the Tkinter event loop
root.mainloop()