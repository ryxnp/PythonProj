import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox

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

# Create the main window
root = tk.Tk()
root.title("Basic GUI Layout")
root.maxsize(1000, 1000)
root.config(bg="grey")

# Create top frame for controls
top_frame = Frame(root, bg='white')
top_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

# Create bottom frame for Treeview and scrollbar
bottom_frame = Frame(root, bg='white')
bottom_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')

# Create StringVars for dropdown selections for the single row
color_var = tk.StringVar()
model_var = tk.StringVar()
sentiment_var = tk.StringVar()

# Create dropdown menu for Color filtering (single row)
Label(top_frame, text="Select Color:").grid(row=0, column=0, padx=5)
color_dropdown = ttk.Combobox(top_frame, textvariable=color_var)
color_dropdown['values'] = ['Select All'] + df['Color'].unique().tolist()  # Add Select All option
color_dropdown.bind('<<ComboboxSelected>>', lambda event: filter_data())
color_dropdown.grid(row=0, column=1, padx=5)

# Create dropdown menu for Model filtering (single row)
Label(top_frame, text="Select Model:").grid(row=0, column=2, padx=5)
model_dropdown = ttk.Combobox(top_frame, textvariable=model_var)
model_dropdown['values'] = ['Select All'] + df['model'].unique().tolist()  # Add Select All option
model_dropdown.bind('<<ComboboxSelected>>', lambda event: filter_data())
model_dropdown.grid(row=0, column=3, padx=5)

# Create dropdown menu for Sentiment filtering (single row)
Label(top_frame, text="Select Sentiment:").grid(row=0, column=4, padx=5)
sentiment_dropdown = ttk.Combobox(top_frame, textvariable=sentiment_var)
sentiment_dropdown['values'] = ['Select All'] + df['Sentiment'].unique().tolist()  # Add Select All option
sentiment_dropdown.bind('<<ComboboxSelected>>', lambda event: filter_data())
sentiment_dropdown.grid(row=0, column=5, padx=5)

# Create a single button named "Analysis"
Button(top_frame, text="Analysis", command=lambda: print("Analysis button clicked")).grid(row=0, column=6, padx=5)

# Centering: Configure grid weights for centering effect in the single row
for i in range(7):
    top_frame.grid_columnconfigure(i, weight=1)  # Allow all columns to expand equally

# Create a Treeview to display the DataFrame in bottom_frame
tree = ttk.Treeview(bottom_frame, columns=list(df.columns), show='headings')
for col in df.columns:
    tree.heading(col, text=col)
    tree.column(col, width=70)  # Set a default width for each column

# Add a scrollbar for horizontal scrolling
scrollbar = Scrollbar(bottom_frame, orient="horizontal", command=tree.xview)
tree.configure(xscrollcommand=scrollbar.set)

# Pack the Treeview and scrollbar into bottom_frame
tree.pack(side='top', fill='both', expand=True)
scrollbar.pack(side='bottom', fill='x')

# Insert initial data into the Treeview
filter_data()  # Show all data initially

# Bind selection event to Treeview
tree.bind("<<TreeviewSelect>>", on_tree_select)

# Configure grid weights to make frames expand properly
root.grid_rowconfigure(0, weight=1)  # Allow top frame to expand vertically
root.grid_rowconfigure(1, weight=3)  # Allow bottom frame to take more space

# Start the Tkinter event loop
root.mainloop()