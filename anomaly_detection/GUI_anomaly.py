import cv2
from datetime import datetime
import logging
import os
import PIL
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.scrolledtext as ScrolledText
from tkinter import ttk, filedialog, PhotoImage
import threading
# Local modules
from raspicam import Raspicam  # Raspberry Pi camera
from text_handler import TextHandler  # Logging to the GUI window
from training_module2 import ModelTrainer  # Model training module
from ttrigger import Trigger, TriggerModes  # Trigger
from ttrigger import PhoneDetector  # Phone detector
import ikony  # Local icons module

Nadpis = "ŠKODA SmartCam"
pad = 5

class App(tk.Tk):
    """
    Main class for the graphical user interface (GUI).
    Contains camera, trigger, model training, and anomaly detection.
    """
    def __init__(self):
        super().__init__()

        # Window properties
        self.title(Nadpis)
        window_scale = 0.9
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        self.geometry(f"{int(width*window_scale)}x{int(height*window_scale)}+{int(width*(1-window_scale)/2)}+{int(height*(1-window_scale)/4)}")
        self.minsize(400, 300)
        self.dataset_path = ""
        self.non_object_path = os.path.dirname(__file__) + "/ar1"  # Folder with non-object images
        self.shots_path = os.path.join(os.path.dirname(__file__), "shots")  # Folder for saving shots
        if not os.path.exists(self.shots_path):
            os.makedirs(self.shots_path)

        self.model_path = ""  # Path to the AI model
        self.trigger_ready = [False, False]  # Flags indicating trigger readiness
        self.trigger_run = False  # Trigger state

        # Camera
        self.camera = Raspicam(use_usb=False)

        # Style settings
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('TButton', foreground='#0e3a2f', relief='flat', font=('verdena', 10, 'bold'))
        self.style.configure('Menu.TLabel', foreground='#0e3a2f', font=('verdena', 12, 'bold'))

        # Initialize GUI widgets
        self.create_widgets()
        self.start_camera_thread()

    def create_widgets(self):
        """Create and configure GUI components."""
        fr = ttk.Frame(self)
        fr.pack(expand=True, fill='both')
        fr.grid_columnconfigure(1, weight=1)
        fr.grid_rowconfigure(0, weight=1)

        self.frOvladani = ttk.Frame(fr, width=200, style='Menu.TFrame')
        self.frOvladani.grid(row=0, column=0, padx=pad, pady=pad, sticky="nsew")

        # Camera settings button
        self.btNastaveni = ttk.Button(self.frOvladani, text="Nastavení kamery", image=self.icon_load("setting"), command=self.openNastaveni, compound=tk.LEFT, width=20)
        self.btNastaveni.grid(row=1, column=0, padx=pad, pady=pad, sticky='nsew')

        # Trigger settings button
        self.btTriggerSettings = ttk.Button(self.frOvladani, text="Nastavení spouštěče", image=self.icon_load("setting"), command=self.open_trigger_settings, compound=tk.LEFT, width=20)
        self.btTriggerSettings.grid(row=2, column=0, padx=pad, pady=pad, sticky='nsew')

        # Save photos and train model buttons
        self.btFunkce1 = ttk.Button(self.frOvladani, text="Ukládání", image=self.icon_load("folder"), command=self.select_trainpicfolder, compound=tk.LEFT, width=20)
        self.btFunkce1.grid(row=3, column=0, padx=pad, pady=(pad / 2, pad / 2), sticky='nsew')

        self.btFunkce2 = ttk.Button(self.frOvladani, text="Foto", image=self.icon_load("camera"), command=self.capture_photo, compound=tk.LEFT, width=20)
        self.btFunkce2.grid(row=4, column=0, padx=pad, pady=(pad / 2, pad / 2), sticky='nsew')
        self.btFunkce2.configure(state="disabled")

        self.btFunkce3 = ttk.Button(self.frOvladani, text="Trénink", image=self.icon_load("start"), command=self.start_training, compound=tk.LEFT, width=20)
        self.btFunkce3.grid(row=5, column=0, padx=pad, pady=(pad / 2, pad / 2), sticky='nsew')
        self.btFunkce3.configure(state="disabled")

        # Model selection and trigger buttons
        self.btFunkce4 = ttk.Button(self.frOvladani, text="Vyber model", image=self.icon_load("model"), command=self.select_model_path, compound=tk.LEFT, width=20)
        self.btFunkce4.grid(row=6, column=0, padx=pad, pady=(2, 2), sticky='nsew')

        self.btFunkce5 = ttk.Button(self.frOvladani, text="Ukládat", image=self.icon_load("folder"), command=self.select_shotsfolder, compound=tk.LEFT, width=20)
        self.btFunkce5.grid(row=7, column=0, padx=pad, pady=(2, 2), sticky='nsew')

        self.btFunkce6 = ttk.Button(self.frOvladani, text="Start Trigger", image=self.icon_load("start"), command=self.start_trigger, compound=tk.LEFT, width=20)
        self.btFunkce6.grid(row=8, column=0, padx=pad, pady=(2, 2), sticky='nsew')
        self.btFunkce6.configure(state="disabled")

        self.btFunkce7 = ttk.Button(self.frOvladani, text="Stop Trigger", image=self.icon_load("stop"), command=self.stop_trigger, compound=tk.LEFT, width=20)
        self.btFunkce7.grid(row=9, column=0, padx=pad, pady=(2, pad), sticky='nsew')
        self.btFunkce7.configure(state="disabled")

        # Logger window
        self.loggerWidget = ScrolledText.ScrolledText(self.frOvladani, wrap='word', state='disabled', width=30, height=10)
        self.loggerWidget.grid(row=10, column=0, padx=pad, pady=(pad / 2, pad / 2), sticky='nsew')
        text_handler = TextHandler(self.loggerWidget)
        logging.basicConfig(filename='test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
        logger.addHandler(text_handler)

        # Camera screen
        self.Imgcanvas = tk.Canvas(fr)
        self.Imgcanvas.grid(row=0, column=1, rowspan=4, padx=pad, pady=pad, sticky="nsew")

    def icon_load(self, icon_name):
        """Load icon from the ikony module."""
        icon_map = {
            "setting": ikony.setting,
            "start": ikony.start,
            "stop": ikony.stop,
            "folder": ikony.folder,
            "model": ikony.model,
            "camera": ikony.camera,
        }
        return PhotoImage(data=icon_map[icon_name])

    def start_camera_thread(self):
        """Start thread for camera video stream."""
        self.bgThread = threading.Thread(target=self.video_stream, daemon=True)
        self.bgThread.start()

    def video_stream(self):
        """Display the video stream from the camera and start phone detection."""
        img = self.camera.capture_img()
        if img is not None:
            if self.trigger_run:
                detector = PhoneDetector(model_path=self.model_path, confidence_threshold=0.53)
                trigger_signal, confidence = detector.detect_phone(img)
                self.trigger.process_trigger_signal(trigger_signal, confidence, img)

            # Update the camera screen
            self.display_image(img)
        self.after(10, self.video_stream)

    def display_image(self, img):
        """Display image on the canvas."""
        image = Image.fromarray(img)
        self.current_img_ref = ImageTk.PhotoImage(image)
        self.Imgcanvas.create_image(0, 0, anchor=tk.NW, image=self.current_img_ref)

    def openNastaveni(self) -> None:
        """Open the camera settings window."""
        self.topLevel = tk.Toplevel(self, background="#dcdad5")
        self.topLevel.title("ŠKODA SmartCam - Nastavení")
        self.topLevel.resizable(False, False)

        ttk.Label(self.topLevel, text="Nastavení kamery", anchor="center").grid(row=0, column=0, padx=pad, pady=pad, columnspan=3, sticky="nsew")

        # Brightness
        ttk.Label(self.topLevel, text="Jas:", anchor='w').grid(row=1, column=0, padx=pad, pady=pad, sticky='nsew')
        ttk.Scale(self.topLevel, from_=30, to=255, variable=self.Bri).grid(row=1, column=1, padx=pad, pady=pad)
        ttk.Label(self.topLevel, textvariable=self.Bri, anchor='w').grid(row=1, column=2, padx=pad, pady=pad, sticky='nsew')

        # Contrast
        ttk.Label(self.topLevel, text="Kontrast:", anchor='w').grid(row=2, column=0, padx=pad, pady=pad, sticky='nsew')
        ttk.Scale(self.topLevel, from_=0, to=10, variable=self.Con).grid(row=2, column=1, padx=pad, pady=pad)
        ttk.Label(self.topLevel, textvariable=self.Con, anchor='w').grid(row=2, column=2, padx=pad, pady=pad, sticky='nsew')

        # Save settings button
        ttk.Button(self.topLevel, text="Uložit", command=self.save_camera_settings).grid(row=3, column=0, columnspan=3, padx=pad, pady=pad, sticky='nsew')

    def open_trigger_settings(self) -> None:
        """Open the trigger settings window."""
        self.trig_window = tk.Toplevel(self, background="#dcdad5")
        self.trig_window.title("ŠKODA SmartCam - Nastavení spouštěče")
        self.trig_window.resizable(False, False)

        ttk.Label(self.trig_window, text="Nastavení spouštěče", anchor="center").grid(row=0, column=0, padx=pad, pady=pad, columnspan=3, sticky="nsew")

        # First delay
        ttk.Label(self.trig_window, text="První zpoždění:", anchor='w').grid(row=1, column=0, padx=pad, pady=pad, sticky='nsew')
        init_delay_entry = ttk.Entry(self.trig_window)
        init_delay_entry.grid(row=1, column=1, padx=pad, pady=pad)
        init_delay_entry.insert(0, self.trigger.trigger_delay)

        # Number of photos
        ttk.Label(self.trig_window, text="Počet fotek:", anchor='w').grid(row=2, column=0, padx=pad, pady=pad, sticky='nsew')
        num_pics_entry = ttk.Entry(self.trig_window)
        num_pics_entry.grid(row=2, column=1, padx=pad, pady=pad)
        num_pics_entry.insert(0, self.trigger.num_of_pictures)

        # Time between photos
        ttk.Label(self.trig_window, text="Čas mezi fotkami:", anchor='w').grid(row=3, column=0, padx=pad, pady=pad, sticky='nsew')
        time_between_pics_entry = ttk.Entry(self.trig_window)
        time_between_pics_entry.grid(row=3, column=1, padx=pad, pady=pad)
        if self.trigger.num_of_pictures > 1:
            time_between_pics_entry.insert(0, self.trigger.times_between_pictures[0])
        else:
            time_between_pics_entry.insert(0, self.trigger.times_between_pictures)

        # Save settings button
        ttk.Button(self.trig_window, text="Uložit", command=lambda: self.save_trigger_vars(
            init_delay_entry.get(), num_pics_entry.get(), time_between_pics_entry.get()
        )).grid(row=4, column=0, columnspan=2, padx=pad, pady=pad, sticky='nsew')

    def select_trainpicfolder(self) -> None:
        """Select folder for saving training images."""
        self.dataset_path = filedialog.askdirectory()
        if self.dataset_path == "" or not os.path.isdir(self.dataset_path):
            logging.info("Složka pro ukládání tréninkových fotek nebyla vybrána.")
        else:
            logging.info(f"Vybraná složka: {self.dataset_path}")
            self.btFunkce2.configure(state="normal")
            self.btFunkce3.configure(state="normal")

    def capture_photo(self) -> None:
        """Capture photo and save it to the selected folder."""
        if not os.path.isdir(self.dataset_path):
            logging.info("Složka pro ukládání tréninkových fotek nebyla nalezena.")
        else:
            self.camera.capture_img_and_save(filename=datetime.now().strftime("%d_%m_%H_%M_%S") + ".png", folder_path=self.dataset_path)
            logging.info("Fotka uložena.")

    def select_model_path(self) -> None:
        """Select model file."""
        self.model_path = filedialog.askopenfilename(title="Vyber model", filetypes=[("Model files", "*.pth")])
        if self.model_path == "" or not os.path.isfile(self.model_path):
            logging.info("Model nebyl vybrán.")
            self.trigger_ready[0] = False
        else:
            logging.info(f"Vybraný model: {self.model_path}")
            self.trigger_ready[0] = True
        self.enable_trigger()

    def select_shotsfolder(self) -> None:
        """Select folder for saving photos."""
        self.shots_path = filedialog.askdirectory()
        if self.shots_path == "" or not os.path.isdir(self.shots_path):
            logging.info("Složka pro ukládání fotek nebyla vybrána.")
            self.trigger_ready[1] = False
        else:
            logging.info(f"Vybraná složka: {self.shots_path}")
            self.trigger_ready[1] = True
            self.trigger = Trigger(None, folder_name=self.shots_path, trigger_delay=5, num_of_pictures=3, times_between_pictures=5)
        self.enable_trigger()

    def start_trigger(self) -> None:
        """Start the trigger."""
        if not os.path.isfile(self.model_path):
            logging.info("Model nebyl vybrán.")
            self.trigger_ready[0] = False
            return
        if not os.path.isdir(self.shots_path):
            logging.info("Složka pro ukládání snímků nebyla vybrána.")
            self.trigger_ready[1] = False
            return

        self.set_all_buttons("disabled")
        self.btFunkce7.configure(state="normal")
        logging.info("Spouštěč spuštěn.")
        self.trigger_run = True

    def stop_trigger(self) -> None:
        """Stop the trigger."""
        logging.info("Spouštěč zastaven.")
        self.set_all_buttons("normal")
        if not os.path.isdir(self.dataset_path):
            self.btFunkce3.configure(state="disabled")
        self.btFunkce7.configure(state="disabled")
        self.trigger_run = False

    def save_camera_settings(self) -> None:
        """Save camera settings."""
        self.camera.set_controls(saturation=self.Sat.get(), sharpness=self.Sha.get(), brightness=self.Bri.get(), contrast=self.Con.get())
        self.save_variables(self.variables_file_path)

    def save_variables(self, path: str) -> None:
        """Save current camera settings to a file."""
        with open(path, 'w') as file:
            file.write(f"Sour={self.Sour.get()}\n")
            file.write(f"Res={self.Res.get()}\n")
            file.write(f"Bri={self.Bri.get()}\n")
            file.write(f"Con={self.Con.get()}\n")
            file.write(f"Fram={self.Fram.get()}\n")
            file.write(f"Exp={self.Exp.get()}\n")
            file.write(f"Sat={self.Sat.get()}\n")
            file.write(f"Sha={self.Sha.get()}\n")

    def save_trigger_vars(self, initial_delay, num_of_pictures, time_between_pictures) -> None:
        """Save trigger settings."""
        self.trigger.set_config(trigger_delay=float(initial_delay), num_of_pictures=int(num_of_pictures), times_between_pictures=float(time_between_pictures))

    def enable_trigger(self) -> None:
        """Enable trigger button if model and shots folder are selected."""
        if self.trigger_ready[0] and self.trigger_ready[1]:
            logging.info("Spouštěč je připraven.")
            self.btFunkce6.configure(state="normal")
        else:
            self.btFunkce6.configure(state="disabled")

    def set_all_buttons(self, state: str) -> None:
        """Set the state of all GUI buttons."""
        if state not in ["normal", "disabled"]:
            raise ValueError("Neplatný stav tlačítka.")
        self.btNastaveni.configure(state=state)
        self.btFunkce1.configure(state=state)
        self.btFunkce2.configure(state=state)
        self.btFunkce3.configure(state=state)
        self.btFunkce4.configure(state=state)
        self.btFunkce5.configure(state=state)
        self.btFunkce6.configure(state=state)
        self.btFunkce7.configure(state=state)

    def start_training(self) -> None:
        """Start training the AI model."""
        logging.info("Trénink spuštěn.")
        self.set_all_buttons("disabled")
        threading.Thread(target=self.run_training, args=(self.dataset_path, self.non_object_path), daemon=True).start()

    def run_training(self, object_folder, non_object_folder) -> None:
        """Function to run AI model training in a separate thread."""
        try:
            trainer = ModelTrainer(object_folder=object_folder, non_object_folder=non_object_folder)
            trainer.train()
            logging.info("Trénink modelu byl úspěšně dokončen.")
        except ValueError as e:
            logging.error(f"Chyba při tréninku modelu: {str(e)}")

    def on_closing(self) -> None:
        """Close the GUI, stop the camera, and save settings."""
        self.save_variables(self.variables_file_path)
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()



