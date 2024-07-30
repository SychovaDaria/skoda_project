import tkinter as tk
from tkinter import messagebox
from threading import Thread

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Trainer")

        self.start_button = tk.Button(root, text="Start Training", command=self.start_training)
        self.start_button.pack(pady=20)

    def start_training(self):
        self.start_button.config(state=tk.DISABLED)
        self.training_thread = Thread(target=self.run_training)
        self.training_thread.start()

    def run_training(self):
        trainer = ModelTrainer(dataset_path='mobil', dataset_path2='ar1', img_height=150, img_width=150, batch_size=32, epochs=30)
        trainer.train()
        messagebox.showinfo("Info", "Training Completed!")
        self.start_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
