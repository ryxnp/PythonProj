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
    main_frame.grid_forget()  # Hide main frame (including Treeview)
    analysis_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')  # Show analysis frame

# Function to go back to the main screen
def show_main_screen():
    analysis_frame.grid_forget()  # Hide analysis frame
    main_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')  # Show main frame again

# Function to display selected analysis graphically
def perform_analysis():
    analysis_type = analysis_var.get()
    x_value = x_var.get()
    y_value = y_var.get()

    if not x_value or not y_value:
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
        
    elif analysis_type == "Pie Chart":
        df[y_value].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie Chart of {y_value}')
        plt.show()
        
    elif analysis_type == "Heatmap":
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True)
        plt.title('Heatmap of Correlation Matrix')
        plt.show()
        
    elif analysis_type == "Violin Plot":
        sns.violinplot(x=x_value, y=y_value, data=df)
        plt.title(f'Violin Plot of {y_value} by {x_value}')
        plt.show()
        
    elif analysis_type == "Pair Plot":
        sns.pairplot(df)
        plt.title('Pair Plot of DataFrame')
        plt.show()

# Create the main window
root = tk.Tk()
root.title("Basic GUI Layout")
root.maxsize(1000, 1000)
root.config(bg="grey")

# Create top frame for controls (main screen)
main_frame = Frame(root, bg='white')
main_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

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
Label(analysis_frame,text="Select Analysis Type:").grid(row=0,column=0,padx=(10))
analysis_var = tk.StringVar()
analysis_dropdown = ttk.Combobox(analysis_frame,textvariable=analysis_var)
analysis_dropdown['values'] = ["Scatter Plot", "Box Plot", "Pie Chart", "Heatmap", "Violin Plot", "Pair Plot"]  # Added more analysis types
analysis_dropdown.grid(row=0,column=1,padx=(10))

Label(analysis_frame,text="Select X Value:").grid(row=0,column=2,padx=(10))
x_var = tk.StringVar()
x_dropdown = ttk.Combobox(analysis_frame,textvariable=x_var)
x_dropdown['values'] = df.columns.tolist()  # Use DataFrame columns as options
x_dropdown.grid(row=0,column=3,padx=(10))

Label(analysis_frame,text="Select Y Value:").grid(row=0,column=4,padx=(10))
y_var = tk.StringVar()
y_dropdown = ttk.Combobox(analysis_frame,textvariable=y_var)
y_dropdown['values'] = df.columns.tolist()  # Use DataFrame columns as options
y_dropdown.grid(row=0,column=5,padx=(10))

Button(analysis_frame,text="Perform Analysis",command=lambda: perform_analysis()).grid(row=0,column=6,padx=(10))

Button(analysis_frame,text="Back",command=lambda: show_main_screen()).grid(row=1,column=6,columnspan=(2),pady=(10), padx=(10))

# Start the Tkinter event loop
root.mainloop()