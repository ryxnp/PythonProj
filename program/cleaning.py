from tkinter import *
import tkinter as tk
from tkinter import ttk
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        df = pd.read_csv('dataset.csv')
        return df.columns.tolist()  # Return list of column names
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

# Function to update the second dropdown based on selection in the first dropdown
def update_dropdown2(*args):
    selected1 = selected_option1.get()
    options2 = [col for col in options if col != selected1]  # Exclude selected option from options2
    selected_option2.set(options2[0] if options2 else '')  # Set default value if available
    dropdown2['values'] = options2  # Update values of second dropdown

# Dropdown menu for selecting use cases
Label(left_frame, text="Select Use Case:").grid(row=1, column=0, padx=5)
use_case_options = ["Plot Squares", "Plot Negatives", "Plot Zeros"]
selected_use_case = StringVar()
selected_use_case.set(use_case_options[0])  # Set default value
use_case_dropdown = ttk.Combobox(left_frame, textvariable=selected_use_case, values=use_case_options)
use_case_dropdown.grid(row=2, column=0, padx=5)

# Dropdown menu for selecting columns from the dataset (X values)
Label(left_frame, text="Select X Column:").grid(row=3, column=0, padx=5)
options = load_dataset()  # Load options from CSV
selected_option1 = StringVar()
if options:
    selected_option1.set(options[0])  # Set default value if options are available
dropdown1 = ttk.Combobox(left_frame, textvariable=selected_option1, values=options)
dropdown1.grid(row=4, column=0, padx=5)
selected_option1.trace('w', update_dropdown2)  # Trace changes in first dropdown

# Second dropdown menu for selecting another column (Y values)
Label(left_frame, text="Select Y Column:").grid(row=5, column=0, padx=5)
selected_option2 = StringVar()
options2 = [col for col in options if col != selected_option1.get()]  # Initial options excluding first selection
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

    # Example data for plotting based on use case selection
    if use_case == "Plot Squares":
        x_values = [1, 2, 3]
        y_values = [i**2 for i in x_values]  # Y is X squared
    elif use_case == "Plot Negatives":
        x_values = [1, 2]
        y_values = [-i for i in x_values]  # Y is negative of X
    elif use_case == "Plot Zeros":
        x_values = [1]
        y_values = [0]  # Y is zero

    # Create a figure and plot data using selected columns if needed (for demonstration purposes)
    fig = Figure(figsize=(5.5, 4))
    ax = fig.add_subplot(111)

    ax.plot(x_values, y_values)  
    ax.set_title(f"Graph using {x_column} (X) and {y_column} (Y)")

    # Create a canvas to display the figure
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

# Button to trigger graph generation
generate_button = Button(left_frame, text="Generate Graph", command=generate_graph)
generate_button.grid(row=7, column=0, padx=5, pady=(10, 5))

root.mainloop()