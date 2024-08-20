import webview
import tkinter as tk

def create_window():
    window = tk.Tk()
    window.geometry("300x200")
    window.title("Hello, world!")
    label = tk.Label(window, text="Hello, world!")
    label.pack(expand=True)
    window.mainloop()

if __name__ == '__main__':
    window = webview.create_window('Hello','http://localhost:5000')
    webview.start(create_window,window)