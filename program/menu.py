import tkinter as tk
from tkinter import ttk
import csv
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dataset Manipulation with Charts")
        self.geometry("800x600")

        # Create a container for the pages
        self.container = ttk.Frame(self)
        self.container.pack(expand=True, fill='both')

        # Configure grid weights for responsiveness
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Initialize the pages
        self.frames = {}
        for F in (MainPage, Page1, Page2):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame("MainPage")

    def show_frame(self, page_name):
        """Show a frame for the given page name."""
        frame = self.frames[page_name]
        frame.tkraise()

class MainPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        # Load dataset
        self.dataset = []
        if not self.load_dataset('iphone.csv'):
            print("Failed to load dataset.")

        # Title Label
        title_label = ttk.Label(self, text="Dataset Manipulation", font=("Helvetica", 16))
        title_label.pack(pady=10)

        # Dataset Info
        info_label = ttk.Label(self, text=f"Entries: {len(self.dataset)}", font=("Helvetica", 12))
        info_label.pack(pady=5)

        if self.dataset:
            headers = ", ".join(self.dataset[0])
            header_label = ttk.Label(self, text=f"Columns: {headers}", font=("Helvetica", 12))
            header_label.pack(pady=5)

        # Entry for number of rows to display
        self.row_entry = tk.Entry(self)
        self.row_entry.pack(pady=10)

        # Button to display sample data
        sample_button = ttk.Button(self, text="Display Sample Data", command=self.display_sample_data)
        sample_button.pack(pady=5)

        # Sample data display area
        self.sample_data_area = tk.Text(self, height=10, width=70)
        self.sample_data_area.pack(pady=10)

        # Navigation buttons
        button1 = ttk.Button(self, text="Go to Page 1", command=lambda: controller.show_frame("Page1"))
        button1.pack(pady=5)

        button2 = ttk.Button(self, text="Go to Page 2", command=lambda: controller.show_frame("Page2"))
        button2.pack(pady=5)

    def load_dataset(self, file_path):
        """Load dataset from a CSV file."""
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                self.dataset = list(csv_reader)  # Load all data into a list
                return True
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return False
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def display_sample_data(self):
        """Display sample data based on user input."""
        try:
            num_rows = int(self.row_entry.get())
            if num_rows > len(self.dataset) - 1:  # Exclude header row
                num_rows = len(self.dataset) - 1
            
            # Format sample data for better readability
            sample_data = ""
            for i, row in enumerate(self.dataset[1:num_rows + 1], start=1):  # Start from 1 to skip header
                formatted_row = f"Row {i}: " + ", ".join(row) + "\n"
                sample_data += formatted_row
            
            self.sample_data_area.delete(1.0, tk.END)  # Clear previous data
            self.sample_data_area.insert(tk.END, sample_data)  # Insert new sample data

        except ValueError:
            self.sample_data_area.delete(1.0, tk.END)
            self.sample_data_area.insert(tk.END, "Please enter a valid number.")

class Page1(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        
        label = ttk.Label(self, text="This is Page 1", font=("Helvetica", 16))
        label.pack(pady=10)

        back_button = ttk.Button(self, text="Back to Main Page", command=lambda: controller.show_frame("MainPage"))
        back_button.pack(pady=5)

class Page2(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        
        label = ttk.Label(self, text="This is Page 2", font=("Helvetica", 16))
        label.pack(pady=10)

        back_button = ttk.Button(self, text="Back to Main Page", command=lambda: controller.show_frame("MainPage"))
        back_button.pack(pady=5)

if __name__ == "__main__":
    app = Application()
    app.mainloop()