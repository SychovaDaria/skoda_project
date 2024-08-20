"""""
Main GUI for Raspberry/USB smart cam v1.1.0


Updates:
    20.6. 2024 - Aded tab view, Variables - loading, saving, handling; Auto creation of settings file
    24.6. 2024 - Modified usage of tkinter variables
    5.7. 2024 - Škoda CI rulles added
    24.7. 2024 - Experiments with multithreading
    14. 8. 2024 - Added comments, docstrings and type hints - JK
    

TODO:
    Add settings for the trigger (f.e. only trigger on rising/falling edge, wait time between triggers, number of pictures
    to take, time between the pictures, etc.)
    Delete commented code
    Rename variables to more descriptive names (And in English)
    Maybe change the trigger to be in a separate thread???

Notes:
    škoda-(vyblednutí 10%) světlezelená -#86fbb6
    škoda-světlezelená - #78faae
    škoda-(tmavší 10%) světlezelená - #6ce19d
    škoda-(nejtmavší 30%) světleězelená -#54af7a
    škoda-(nejtmavší 50%) světleězelená - #3c7d57

    škoda- (vyblednutí10%) tmavězelená -#264e44
    škoda-tmavězelená - #0e3a2f
    škoda-(tmavší10%) tmavězelená - #0d342a
    škoda-(nejtmavší 50%) tmavězelená - #0a2921
"""""

# standard
import cv2
from datetime import datetime
import logging
import os
import PIL
import PIL.Image
import PIL.ImageTk
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.scrolledtext as ScrolledText
from tkinter import ttk
from tkinter import filedialog, PhotoImage
import threading
# local (our modules)
from raspicam import Raspicam  # TODO: change to import the raspicam module
from text_handler import TextHandler
from training_module2 import ModelTrainer
from trigger import Trigger
from trigger_module3 import PhoneDetector


Nadpis = "ŠKODA SmartCam"
pad = 10


##JP - trochu jsem změnil a dal do if-else
default_json = os.path.join(os.path.dirname(__file__),'skodaCI.json')
if os.path.exists(default_json):
    #ttk.set_default_color_theme(default_json)
    pass
else:
    print(f"UPOZORNENI: Soubor skodaCI.json nebyl nalezen ")

#ttk.set_appearance_mode("dark")


class App(tk.Tk):
    """
    Class for the main GUI application.

    Creates a GUI window created in Tkinter, consisting of buttons, logging text window and a camera stream. The user can adjust the 
    camera settings, set the path for saving the training images, start training the AI, choose the AI model, start and stop the picture 
    aquisition (trigger), and set multiple settings regarding the trigger.
    The GUI also contains a logging window for displaying the application's log.

    Attributes:
        dataset_path (str): Path to the folder where the training images are saved.
        non_object_path (str): Path to the folder with non-object images.
        model_path (str): Path to the AI model.
        shots_path (str): Path to the folder where the trigger will save
        trigger_ready (list): List of two booleans, indicating if the model and the shots path are selected.
        trigger_run (bool): Boolean indicating if the trigger is running. #TODO: can maybe delete????
        camera (Raspicam): Camera object.
        Sour (tk.StringVar): Variable for camera source.
        Res (tk.StringVar): Variable for camera resolution.
        Bri (tk.DoubleVar): Variable for camera brightness.
        Con (tk.DoubleVar): Variable for camera contrast.
        Fram (tk.DoubleVar): Variable for camera framerate.
        Exp (tk.DoubleVar): Variable for camera exposure.
        Sat (tk.DoubleVar): Variable for camera saturation.
        Sha (tk.DoubleVar): Variable for camera sharpness.
        variables_file_path (str): Path to the file with the variables.
        style (ttk.Style): Style of the GUI.
        Frames and widgets of the GUI. (not listed for brevity)
    """
    def __init__(self):
        super().__init__()

        # Window properties
        window_scale = 0.9 ##JP - % z obrazovky
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        self.geometry(f"{int(width*window_scale)}x{int(height*window_scale)}+{int(width*(1-window_scale)/2)}+{int(height*(1-window_scale)/4)}")  ##JP - vycentroval jsem a používám měřítko window_scale
        self.minsize(400, 300)
        self.title(Nadpis)
        self.dataset_path=""
        self.non_object_path=os.path.dirname(__file__) + "../v1.0/ar1" # the folder with non-object images FIXME: didnt want to copy the img folder, so i changed the path momentarily
        self.style = ttk.Style(self)
        self.style.theme_use('vista')

        #self.style.configure('TFrame', background='#1a1a1a')
        self.style.configure('Menu.TFrame', background='#0e3a2f')
        #self.style.configure('TLabel', background='#1a1a1a', foreground='#0e3a2f', font=('verdena',12,'bold'))
        self.style.configure('TButton', background='#0e3a2f', foreground='#6ce19d', relief='flat', font=('verdena',10,'bold'))
        self.style.configure('Menu.TLabel', background='#0e3a2f', foreground='#78faae', font=('verdena',12,'bold'))
        #self.style.configure('TNotebook', background='#1a1a1a',tabposition='n', darkcolor='white', relief='flat')
        #self.style.configure('TNotebook.Tab', background='white', foreground='#78faae', font=('verdena',10,'bold'), tabposition='n')
        #self.style.configure('TMenubutton', background='white', foreground='#78faae', font=('verdena',10,'bold'), relief='flat')

        #self.style.map('TButton',background=[('active', '#91ffd4')], relief=[('active', 'flat')])

        #self.icon_setting=PhotoImage(data=ikony.setting)
        #self.icon_start=PhotoImage(data=ikony.start)
        #self.icon_stop=PhotoImage(data=ikony.stop)
        #self.icon_folder=PhotoImage(data=ikony.folder)
        #self.icon_model=PhotoImage(data=ikony.model)
        #self.icon_camera=PhotoImage(data=ikony.camera)
        #self.skoda_logo=PhotoImage(data=ikony.logo)

        # Initialize tkinter variables
        self.Sour = tk.StringVar()
        self.Res = tk.StringVar()
        self.Bri = tk.DoubleVar()
        self.Con = tk.DoubleVar()
        self.Fram = tk.DoubleVar()
        self.Exp = tk.DoubleVar()
        self.Sat = tk.DoubleVar()
        self.Sha = tk.DoubleVar()

        

        # Loading variables at start
        self.variables_file_path = os.path.join(os.path.dirname(__file__),"values.txt")
        try:
            self.load_variables(self.variables_file_path)
        except:
            raise Exception("CHYBA: Chyba při načítání proměnných")

        self.camera = Raspicam(use_usb=True) # start the camera
        try:
            self.nactiGUI()
        except:
            raise Exception("CHYBA: Chyba při načítání GUI")

        self.bgThread = threading.Thread(target=self.video_stream, daemon=True) # thread for the video stream
        self.bgThread.start()
        self.trigger_run = False
        self.trigger = Trigger(None, trigger_delay=5, num_of_pictures= 3, times_between_pictures= 5)
        logging.info("Aplikace spuštěna")
        
        

    # GUI init
    def nactiGUI(self) -> None:
        """
        Loads the GUI and its components.

        Args:
            None

        Returns:
            None
        """
        # Main frame
        fr = ttk.Frame(self)
        fr.pack(expand=True, fill='both')
        fr.grid_columnconfigure(1, weight=1)
        fr.grid_rowconfigure(0, weight=1)

        # Frames
        self.frOvladani = ttk.Frame(fr, width=227, style='Menu.TFrame')
        self.frOvladani.grid(row=0, column=0, padx=pad, pady=pad, sticky="nsew")

        self.frTrenink = ttk.Frame(self.frOvladani, style='Menu.TFrame')#, fg_color="#264e44")
        self.frTrenink.grid(row=2, column=0, padx=pad, pady=pad, sticky="nsew")

        self.frVyhodnoceni = ttk.Frame(self.frOvladani, style='Menu.TFrame')#, fg_color="#264e44")
        self.frVyhodnoceni.grid(row=3, column=0, padx=pad, pady=pad, sticky="nsew")

        self.frLogger = ttk.Frame(self.frOvladani, style='Menu.TFrame')#, fg_color="#264e44")
        self.frLogger.grid(row=4, column=0, padx=pad, pady=pad, sticky="nsew")

        self.Imgcanvas= tk.Canvas (fr)
        self.Imgcanvas.grid(row=0, column=1, rowspan=4, padx=pad, pady=(pad), sticky="nsew")
        

        # Widgets - Left frame

        #self.logo_label = ttk.Label(self.frOvladani, text="", image =self.skoda_logo, style='Menu.TLabel')
        #self.logo_label.grid(row=0, column=0, padx=pad, pady=pad, sticky="nsew")
        # Buttons
        # camera settings
        self.btNastaveni = ttk.Button(self.frOvladani, text="Nastavení kamery",  command=self.openNastaveni, compound=tk.LEFT,width=20, style='TButton')
        self.btNastaveni.grid(row=1, column=0, padx=pad, pady=pad, sticky='nsew')

        # Training
        self.training_label = ttk.Label(self.frTrenink, text="Trénování modelu", style='Menu.TLabel')#, corner_radius= 6, fg_color="#78faae", text_color="#0e3a2f", font=ttk.Font(weight="bold"))
        self.training_label.grid(row=0, column=0, padx=pad/2, pady= (15,20), sticky="nsew")

        self.btFunkce1 = ttk.Button(self.frTrenink, text="Ukládání", command=self.selectTrainpicfolder, compound=tk.LEFT,width=20)
        self.btFunkce1.grid(row=1, column=0, padx=pad, pady=(pad/2,pad/2), sticky='nsew')

        self.btFunkce2 = ttk.Button(self.frTrenink, text="Foto", command=self.capturephoto, compound=tk.LEFT,width=20)
        self.btFunkce2.grid(row=2, column=0, padx=pad, pady=(pad/2,pad/2), sticky='nsew')
        self.btFunkce2.configure(state="disabled")

        self.btFunkce3 = ttk.Button(self.frTrenink, text="Trénink", command=self.start_training, compound=tk.LEFT,width=20)
        self.btFunkce3.grid(row=3, column=0, padx=pad, pady=(pad/2,pad/2), sticky='nsew')
        self.btFunkce3.configure(state="disabled")

        # Evaluation and trigger
        self.evaluation_label = ttk.Label(self.frVyhodnoceni, text="Použití modelu", style='Menu.TLabel')#, corner_radius= 6, fg_color="#78faae", text_color="#0e3a2f", font=ttk.Font(weight="bold"))
        self.evaluation_label.grid(row=0, column=0, padx=pad/2, pady= (15,20), sticky="nsew")

        self.btFunkce4 = ttk.Button(self.frVyhodnoceni, text="Model", command=self.selectmodelpath, compound=tk.LEFT,width=20)
        self.btFunkce4.grid(row=1, column=0, padx=pad, pady=(2,2), sticky='nsew')

        self.btFunkce5 = ttk.Button(self.frVyhodnoceni, text="Ukládat",  command=self.selectshotsfolder, compound=tk.LEFT,width=20)
        self.btFunkce5.grid(row=2, column=0, padx=pad, pady=(2,2), sticky='nsew')

        self.trigger_ready = [False, False]
        self.btFunkce6 = ttk.Button(self.frVyhodnoceni, text="Start Trigger", command=self.start_trigger, compound=tk.LEFT,width=20)
        self.btFunkce6.grid(row=3, column=0, padx=pad, pady=(2,2), sticky='nsew')
        self.btFunkce6.configure(state="disabled")

        self.btFunkce7 = ttk.Button(self.frVyhodnoceni, text="Stop Trigger", command=self.stop_trigger, compound=tk.LEFT,width=20)
        self.btFunkce7.grid(row=4, column=0, padx=pad, pady=(2,pad), sticky='nsew')
        self.btFunkce7.configure(state="disabled")

        # logger text widget
        self.logger_label = ttk.Label(self.frLogger, text="Logger", style='Menu.TLabel')#, corner_radius= 6, fg_color="#78faae", text_color="#0e3a2f", font=ttk.Font(weight="bold"))
        self.logger_label.grid(row=0, column=0, padx=pad/2, pady= (15,20), sticky="nsew")

        self.loggerWidget = ScrolledText.ScrolledText(self.frLogger, wrap='word', state='disabled', width=30, height=10)
        self.loggerWidget.grid(row=1, column=0, padx=pad, pady=(pad/2,pad/2), sticky='nsew')
        text_handler = TextHandler(self.loggerWidget)
        logging.basicConfig(filename='test.log',
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s')  
        
        logger = logging.getLogger()
        logger.addHandler(text_handler)
        
    # Settings window
    def openNastaveni(self) -> None:
        """
        Opens the settings window for the camera.

        Args:
            None
        
        Returns:
            None
        """
        self.topLevel = tk.Toplevel(self,)
        self.topLevel.title("ŠKODA SmartCam - Nastavení")
        self.topLevel.resizable(False, False)

        # Nadpis
        ttk.Label(self.topLevel, text="Nastavení kamery", anchor="center").grid(row=0, column=0, padx=pad, pady=pad, columnspan=3, sticky="nsew")

        # Settings Item
        # Source
        ttk.Label(self.topLevel, text="Zdroj:",anchor='w').grid(row=1, column=0, padx=pad, pady=pad, sticky='nsew')

        ttk.OptionMenu(self.topLevel, self.Sour, self.Sour.get(),*["RasPi", "USB"], direction='below').grid(row=1, column=1, padx=pad, pady=pad)

        # Resolution
        ttk.Label(self.topLevel, text="Rozlišení:", anchor='w').grid(row=2, column=0, padx=pad, pady=pad, sticky='nsew')

        ttk.OptionMenu(self.topLevel, self.Res, self.Res.get(), *["1920x1080","1280x720", "640x480"]).grid(row=2, column=1, padx=pad, pady=pad)

        # Brightness
        ttk.Label(self.topLevel, text="Jas:", anchor='w').grid(row=3, column=0, padx=pad, pady=pad, sticky='nsew')

        ttk.Scale(self.topLevel, from_=30, to=255, variable=self.Bri).grid(row=3, column=1, padx=pad, pady=pad)
        ttk.Label(self.topLevel, textvariable=self.Bri, anchor='w').grid(row=3, column=2, padx=pad, pady=pad, sticky='nsew')

        # Contrast
        ttk.Label(self.topLevel, text="Kontrast:", anchor='w').grid(row=4, column=0, padx=pad, pady=pad, sticky='nsew')

        ttk.Scale(self.topLevel, from_=0, to=10, variable=self.Con).grid(row=4, column=1, padx=pad, pady=pad)
        ttk.Label(self.topLevel, textvariable=self.Con, anchor='w').grid(row=4, column=2, padx=pad, pady=pad, sticky='nsew')

        # Framerate
        ttk.Label(self.topLevel, text="Framerate:", anchor='w').grid(row=5, column=0, padx=pad, pady=pad, sticky='nsew')

        ttk.Scale(self.topLevel, from_=-0.8, to=0.8, variable=self.Fram).grid(row=5, column=1, padx=pad, pady=pad)
        ttk.Label(self.topLevel, textvariable=self.Fram, anchor='w').grid(row=5, column=2, padx=pad, pady=pad, sticky='nsew')

        # Exposure
        ttk.Label(self.topLevel, text="Expozice:", anchor='w').grid(row=6, column=0, padx=pad, pady=pad, sticky='nsew')

        ttk.Scale(self.topLevel, from_=0, to=100, variable=self.Exp).grid(row=6, column=1, padx=pad, pady=pad)
        ttk.Label(self.topLevel, textvariable=self.Exp, anchor='w').grid(row=6, column=2, padx=pad, pady=pad, sticky='nsew')

        # Saturation
        ttk.Label(self.topLevel, text="Saturace:", anchor='w').grid(row=7, column=0, padx=pad, pady=pad, sticky='nsew')

        ttk.Scale(self.topLevel, from_=0, to=200, variable=self.Sat).grid(row=7, column=1, padx=pad, pady=pad)
        ttk.Label(self.topLevel, textvariable=self.Sat, anchor='w').grid(row=7, column=2, padx=pad, pady=pad, sticky='nsew')

        # Sharpness
        ttk.Label(self.topLevel, text="Ostrost:", anchor='w').grid(row=8, column=0, padx=pad, pady=pad, sticky='nsew')

        ttk.Scale(self.topLevel, from_=0, to=50, variable=self.Sha).grid(row=8, column=1, padx=pad, pady=pad)
        ttk.Label(self.topLevel, textvariable=self.Sha, anchor='w').grid(row=8, column=2, padx=pad, pady=pad, sticky='nsew')

        #Definition of  Doublevar tracing - for rounding them 
        self.topLevel.attributes('-topmost', 'true')
        self.Bri.trace_add("write", lambda *args: (self.round_and_update_var(self.Bri), self.camera.set_controls(brightness=self.Bri.get())))
        self.Con.trace_add("write", lambda *args: (self.round_and_update_var(self.Con), self.camera.set_controls(contrast=self.Con.get())))
        self.Fram.trace_add("write", lambda *args: self.round_and_update_var(self.Fram))
        self.Exp.trace_add("write", lambda *args: self.round_and_update_var(self.Exp))
        self.Sat.trace_add("write", lambda *args: (self.round_and_update_var(self.Sat), self.camera.set_controls(saturation=self.Sat.get())))
        self.Sha.trace_add("write", lambda *args: (self.round_and_update_var(self.Sha), self.camera.set_controls(sharpness=self.Sha.get())))
        self.Res.trace_add("write", lambda *args: self.set_resolution())
        

    # Loading of variables values
    def load_variables(self, path: str) -> None:
        """
        Loads the camera settings from the file.
        If the file does not exist, sets the default values.

        Args:
            path (str): Path to the file with the variables.
        
        Returns:
            None
        """
        default_values = {"Sour": "USB", "Res": "640x480", "Bri": 1, "Con": 6, "Fram": 0.8, "Exp": 0.8, "Sat": 0.8, "Sha": 0.8}

        if os.path.exists(path):
            with open(path, 'r') as file:
                data = file.readlines()
                data_dict = {}
                for line in data:
                    if '=' in line:
                        key, value = line.strip().split('=')
                        data_dict[key.strip()] = value.strip()
                self.Sour.set(data_dict.get("Sour", default_values["Sour"]))
                self.Res.set(data_dict.get("Res", default_values["Res"]))
                self.Bri.set(float(data_dict.get("Bri", default_values["Bri"])))
                self.Con.set(float(data_dict.get("Con", default_values["Con"])))
                self.Fram.set(float(data_dict.get("Fram", default_values["Fram"])))
                self.Exp.set(float(data_dict.get("Exp", default_values["Exp"])))
                self.Sat.set(float(data_dict.get("Sat", default_values["Sat"])))
                self.Sha.set(float(data_dict.get("Sha", default_values["Sha"])))
        else:
            self.Sour.set(default_values["Sour"])
            self.Res.set(default_values["Res"])
            self.Bri.set(default_values["Bri"])
            self.Con.set(default_values["Con"])
            self.Fram.set(default_values["Fram"])
            self.Exp.set(default_values["Exp"])
            self.Sat.set(default_values["Sat"])
            self.Sha.set(default_values["Sha"])
            self.save_variables(path)


    # Saving variables values
    def save_variables(self, path: str) -> None:
        """
        Saves the current camera settings to the file.

        Args:
            path (str): Path to the file with the variables.
        
        Returns:
            None
        """
        with open(path, 'w') as file:
            file.write(f"Sour={self.Sour.get()}\n")
            file.write(f"Res={self.Res.get()}\n")
            file.write(f"Bri={self.Bri.get()}\n")
            file.write(f"Con={self.Con.get()}\n")
            file.write(f"Fram={self.Fram.get()}\n")
            file.write(f"Exp={self.Exp.get()}\n")
            file.write(f"Sat={self.Sat.get()}\n")
            file.write(f"Sha={self.Sha.get()}\n")

    #Funtion for rounding of  DoubleVar
    def round_and_update_var(self, var): #??????
        # Round the DoubleVar value to one decimal place
        rounded_value = round(var.get(), 1)
        # Set the rounded value back to the DoubleVar
        var.set(rounded_value)

    def set_resolution(self) -> None:
        """
        Sets the camera resolution to the selected value.

        Args:
            None

        Returns:
            None
        """
        resolution_str=self.Res.get()
        width, height = map(int, resolution_str.split('x'))
        self.camera.change_resolution((width,height))
        logging.info(f"Rozlišení kamery změněno na {resolution_str}")

    def video_stream(self) -> None:
        """
        Updates the canvas widget with the camera stream.
        Updates the camera stream in GUI, also calls the AI detector if trigger was started. Is called by own thread.

        Args:
            None
        
        Returns:
            None
        """
        img = self.camera.capture_img()
        
        if img is not None:
            if self.trigger_run:
                detector = PhoneDetector(model_path=self.model_path,confidence_threshold=0.5)
                self.trigger.process_trigger_signal(detector.detect_phone(img), img)
            image = Image.fromarray(img)
            self.current_img_ref= PIL.ImageTk.PhotoImage(image)
            self.backround_img = self.Imgcanvas.create_image((self.Imgcanvas.winfo_width()/2),(self.Imgcanvas.winfo_height()/2), image=self.current_img_ref)
        img=cv2.resize(img, (self.Imgcanvas.winfo_width(), self.Imgcanvas.winfo_height()))
        self.current_img_ref=PIL.ImageTk.PhotoImage(PIL.Image.fromarray(img))
        self.Imgcanvas.itemconfigure(self.backround_img, image=self.current_img_ref)

        self.video=self.after(10, self.video_stream) # loop

    def selectTrainpicfolder(self) -> None:
        """
        Selects and saves the folder for the training images.

        Args:
            None
        
        Returns:
            None        
        """
        self.dataset_path=filedialog.askdirectory()
        if self.dataset_path == "" or not os.path.isdir(self.dataset_path): # check if the folder is selected
            logging.info("Složka pro ukládání trénovacích fotek nebyla vybrána")
        else:
            logging.info(f"Vybrána složka pro ukládání trénovacích fotek: {self.dataset_path}")
            # set the foto buttons and train buttons active
            self.btFunkce2.configure(state="normal")
            self.btFunkce3.configure(state="normal")

    def capturephoto(self) -> None:
        """
        Function for manually capturing the photos.
        Saves the photo in .png format to the previously selected training folder, with the current date and time as the filename.

        Args:
            None

        Returns:
            None
        """
        if not os.path.isdir(self.dataset_path): # check if the folder still exists
            logging.info("Složka pro ukládání trénovacích fotek nebyla nalezena")
        else:
            self.camera.capture_img_and_save(filename=datetime.now().strftime("%d_%m_%H_%M_%S") + ".png", folder_path=self.dataset_path)
            logging.info("Fotka uložena")

    def selectmodelpath(self) -> None:
        """
        Selects the file with the AI model.

        Args:
            None

        Returns:
            None
        """
        self.model_path=filedialog.askopenfilename(title="Vyber model", filetypes=[("Model files", "*.pth")])
        if self.model_path == "" or not os.path.isfile(self.model_path): # check if the folder is selected
            logging.info("Model nebyl vybrán")
            self.trigger_ready[0] = False
        else:
            logging.info(f"Vybrán model ze souboru: {self.model_path}")
            self.trigger_ready[0] = True
            self.enable_trigger()

    def selectshotsfolder(self) -> None:
        """
        Selects the folder for saving the aquired images.

        Args:
            None

        Returns:
            None
        """
        self.shots_path=filedialog.askdirectory()
        if self.shots_path == "" or not os.path.isdir(self.shots_path): # check if the folder is selected
            logging.info("Složka nebyla vybrána")
            self.trigger_ready[1] = False
        else:
            logging.info(f"Vybrána složka pro ukládání fotek: {self.shots_path}")
            self.trigger_ready[1] = True
            self.trigger.set_config(folder_name=self.shots_path)
            self.enable_trigger()

    def enable_trigger(self) -> None:
        """
        Enables the trigger button if both the model and the folder for shots are selected.

        Args:
            None

        Returns:
            None
        """
        #print(self.trigger_ready)
        if self.trigger_ready[0] and self.trigger_ready[1]:
            logging.info("Trigger je připraven")
            self.btFunkce6.configure(state="normal")

    def start_trigger(self) -> None:
        """
        Starts the AI trigger (image aquisition).
        """
        # check if both the model and the folder for shots are selected
        if not os.path.isfile(self.model_path):
            logging.info("Model nebyl vybrán")
            self.trigger_ready[0] = False
            return
        if not os.path.isdir(self.shots_path):
            logging.info("Složka pro ukládání snímků nebyla vybrána")
            self.trigger_ready[1] = False
            return
        #self.tabview.set("Blank")
        self.set_all_buttons("disabled")
        self.btFunkce7.configure(state="normal")
        logging.info("Trigger spuštěn")
        self.trigger_run = True
        
    def start_training(self) -> None:
        """
        Starts the training of the AI model.

        Args:
            None

        Returns:
            None
        """
        logging.info("Trénink spuštěn")
        self.set_all_buttons("disabled")
        threading.Thread(target=self.run_training, args=(self.dataset_path, self.non_object_path), daemon=True).start()

    def run_training(self, object_folder, non_object_folder) -> None:
        """
        Function for running the training of the AI model, is called by a separate thread.

        Args:
            object_folder (str): Path to the folder with the object images.
            non_object_folder (str): Path to the folder with the non-object images.

        Returns:
            None
        """
        try:
            self.trainer = ModelTrainer(object_folder=object_folder, non_object_folder=non_object_folder)
            self.trainer.train()
            #ttk.messagebox.showinfo("Training", "Model training has started.")
        except ValueError as e:
            #ttk.messagebox.showerror("Error", str(e))
            logging.info("Chyba při tréninku modelu")

    def set_all_buttons(self, state:str)->None:
        """
        Sets the state of all buttons in the GUI to be either normal or disabled.

        Args:
            state (str): State to set the buttons to. Can be either "normal" or "disabled".

        Returns:
            None
        """
        if state!="normal" and state!="disabled":
            raise ValueError("Neplatný stav tlačítka")
        self.btNastaveni.configure(state=state)
        self.btFunkce1.configure(state=state)
        self.btFunkce2.configure(state=state)
        self.btFunkce3.configure(state=state)
        self.btFunkce4.configure(state=state)
        self.btFunkce5.configure(state=state)
        self.btFunkce6.configure(state=state)
        self.btFunkce7.configure(state=state)

    def stop_trigger(self) -> None:
        """
        Stops the trigger.

        Args:
            None

        Returns:
            None
        """
        logging.info("Trigger zastaven")
        self.set_all_buttons("normal")
        if not os.path.isdir(self.dataset_path):
            self.btFunkce3.configure(state="disabled")
        self.btFunkce7.configure(state="disabled")
        self.trigger_run = False


    def on_closing(self) -> None:
        """
        Closes the GUI, stops the camera and saves the settings variables into a file.

        Args:
            None

        Returns:
            None
        """
        self.save_variables(self.variables_file_path)
        self.destroy()

if __name__ == "__main__":
    # Application start
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
    