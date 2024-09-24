import tkinter as tk
from tkinter import ttk
import subprocess  # Для запуска других приложений

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Výběr režimu práce")
        self.geometry("400x200")

        # Label to prompt the user to select a mode
        label = ttk.Label(self, text="Vyberte režim:", font=("Helvetica", 14))
        label.pack(pady=20)

        # Button to open the trigger mode window (launch GUI2.py)
        trigger_button = ttk.Button(self, text="Trigger", command=self.open_trigger_app)
        trigger_button.pack(pady=10)

        # Button to open the anomaly detector mode window (launch GUI_anomaly.py)
        anomaly_button = ttk.Button(self, text="Detektor Anomálií", command=self.open_anomaly_app)
        anomaly_button.pack(pady=10)

    # Function to open the trigger application (runs GUI2.py)
    def open_trigger_app(self):
        self.withdraw()  # Hide the main window
        subprocess.Popen(["python", "GUI2.py"])  # Запускает GUI2.py как отдельный процесс
        self.destroy()  # Закрывает главное окно

    # Function to open the anomaly detector application (runs GUI_anomaly.py)
    def open_anomaly_app(self):
        self.withdraw()  # Hide the main window
        subprocess.Popen(["python", "GUI_anomaly.py"])  # Запускает GUI_anomaly.py как отдельный процесс
        self.destroy()  # Закрывает главное окно

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
