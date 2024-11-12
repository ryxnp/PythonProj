import tkinter as tk

def on_button_click():
    user_input = entry.get()
    label.config(text=f"You entered: {user_input}")

# Create the main window
root = tk.Tk()
root.title("Simple Tkinter Layout")
root.geometry("300x200")

# Create a label
label = tk.Label(root, text="Enter something:")
label.pack(pady=10)

# Create an entry widget
entry = tk.Entry(root)
entry.pack(pady=10)

# Create a button
button = tk.Button(root, text="Submit", command=on_button_click)
button.pack(pady=10)

# Run the application
root.mainloop()