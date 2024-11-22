from tkinter import *
import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns  # For heatmap and boxplot
import numpy as np  # For generating random data

# Create root window
root = Tk()
root.title("Basic GUI Layout")
root.maxsize(900, 600)
root.config(bg="grey")

# Create left and right frames
left_frame = Frame(root, width=200, height=400, bg='white')
left_frame.grid(row=0, column=0, padx=10, pady=5)

right_frame = Frame(root, width=650, height=400, bg='white')
right_frame.grid(row=0, column=1, padx=10, pady=5)

# Create frames and labels in left_frame
Label(left_frame, text="Menu Controls", font=("Arial", 14)).grid(row=0, column=0, padx=5, pady=5)

# Load dataset and populate dropdown menu with column names
def load_dataset():
    try:
        # Load CSV file into a DataFrame
        df = pd.read_csv('processed_dataset.csv')
        return df  # Return the DataFrame itself for processing
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Function to update the second dropdown based on selection in the first dropdown
def update_dropdown2(*args):
    selected1 = selected_option1.get()
    options2 = [col for col in options.columns if col != selected1]  # Exclude selected option from options2
    selected_option2.set(options2[0] if options2 else '')  # Set default value if available
    dropdown2['values'] = options2  # Update values of second dropdown

# Dropdown menu for selecting use cases
Label(left_frame, text="Select Use Case:").grid(row=1, column=0, padx=5)
use_case_options = ["Scatter Plot", "Pie Chart", "Box Plot", "HeatMap"]
selected_use_case = StringVar()
selected_use_case.set(use_case_options[0])  # Set default value
use_case_dropdown = ttk.Combobox(left_frame, textvariable=selected_use_case, values=use_case_options)
use_case_dropdown.grid(row=2, column=0, padx=5)

# Load dataset and create dropdown menus for selecting columns from the dataset (X and Y values)
options = load_dataset()  # Load options from CSV
Label(left_frame, text="Select X Column:").grid(row=3, column=0, padx=5)
selected_option1 = StringVar()
if not options.empty:
    selected_option1.set(options.columns[0])  # Set default value if options are available
dropdown1 = ttk.Combobox(left_frame, textvariable=selected_option1, values=list(options.columns))
dropdown1.grid(row=4, column=0, padx=5)
selected_option1.trace('w', update_dropdown2)  # Trace changes in first dropdown

# Second dropdown menu for selecting another column (Y values)
Label(left_frame, text="Select Y Column:").grid(row=5, column=0, padx=5)
selected_option2 = StringVar()
options2 = [col for col in options.columns if col != selected_option1.get()]  # Initial options excluding first selection
selected_option2.set(options2[0] if options2 else '')  # Set default value if available
dropdown2 = ttk.Combobox(left_frame, textvariable=selected_option2)
dropdown2.grid(row=6, column=0, padx=5)

# Function to validate input and generate graph/table based on it
def generate_graph():
    # Clear existing figure if any
    for widget in right_frame.winfo_children():
        widget.destroy()

    # Get selected options from dropdowns
    x_column = selected_option1.get()
    y_column = selected_option2.get()
    use_case = selected_use_case.get()

    # Extract X and Y values from the DataFrame based on user selection
    x_values = options[x_column].dropna().values  # Drop NaN values for X
    y_values = options[y_column].dropna().values  # Drop NaN values for Y

    fig = Figure(figsize=(5.5, 4))
    ax = fig.add_subplot(111)

    if use_case == "Scatter Plot":
        ax.scatter(x_values[:50], y_values[:50])  # Scatter plot with limited points
    
    elif use_case == "Pie Chart":
        sizes = [np.count_nonzero(options[y_column] == val) for val in np.unique(options[y_column])]
        labels = np.unique(options[y_column])
        ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    
    elif use_case == "Box Plot":
        data_to_plot = [options[x_column].dropna(), options[y_column].dropna()]  # Ensure no NaN values are plotted
        ax.boxplot(data_to_plot)
    
    elif use_case == "HeatMap":
        heatmap_data = pd.DataFrame(np.random.rand(10, len(options.columns)), columns=list(options.columns))
        sns.heatmap(heatmap_data.corr(), ax=ax) 

    ax.set_title(f"Graph using {x_column} (X) and {y_column} (Y)")
    
    # Create a canvas to display the figure
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

# Function to display dataset in a text widget
def process_data():
    for widget in right_frame.winfo_children():
        widget.destroy()  # Clear existing widgets

    df_text_area = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD)
    df_text_area.pack(fill=BOTH, expand=True)

    df_text_area.insert(tk.END, str(options))  # Display the DataFrame as text

# Button to trigger graph generation
generate_button = Button(left_frame, text="Generate Graph", command=generate_graph)
generate_button.grid(row=7,column=0,padx=5,pady=1)

# Button to process data and display the DataFrame
process_button = Button(left_frame,text="Display Dataset",command=process_data)
process_button.grid(row=8,column=0,padx=5,pady=3)

root.mainloop()