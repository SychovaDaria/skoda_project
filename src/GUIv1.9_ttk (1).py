"""""
GUI for Raspberry/USB smart cam v.1.6
Main window + settings window
20.6. 2024 - Aded tab view, Variables - loading, saving, handling; Auto creation of settings file
24.6. 2024 - Modified usage of tkinter variables
5.7. 2024 - Škoda CI rulles added
24.7. 2024 - Experiments with multithreading

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


import PIL.Image
#import customtkinter as 
import tkinter as tk
from tkinter import ttk
import PIL
import os
from PIL import Image, ImageTk
import PIL.ImageTk
from camera_module import Raspicam  # Assuming the camera module code is saved as camera_module.py
from tkinter import filedialog, PhotoImage
import threading
#import ikony  ##JP - ikony jsem přesunul do samostatnýho souboru
from datetime import datetime
import cv2

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
        self.dataset2_path=os.path.dirname(__file__) + "/ar1"
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

        self.camera = Raspicam()
        #self.camera = cv2.VideoCapture(0)
        try:
            self.nactiGUI()
        except:
            raise Exception("CHYBA: Chyba při načítání GUI")

        bgThread = threading.Thread(target=self.video_stream, daemon=True)
        bgThread.start()
        
        

    # GUI init
    def nactiGUI(self):
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

        self.Imgcanvas= tk.Canvas (fr)
        self.Imgcanvas.grid(row=0, column=1, rowspan=4, padx=pad, pady=(pad), sticky="nsew")
        

        # Widgets - Left frame

        #self.logo_label = ttk.Label(self.frOvladani, text="", image =self.skoda_logo, style='Menu.TLabel')
        #self.logo_label.grid(row=0, column=0, padx=pad, pady=pad, sticky="nsew")

        self.btNastaveni = ttk.Button(self.frOvladani, text="Nastavení kamery",  command=self.openNastaveni, compound=tk.LEFT,width=20, style='TButton')
        self.btNastaveni.grid(row=1, column=0, padx=pad, pady=pad, sticky='nsew')

        self.training_label = ttk.Label(self.frTrenink, text="Trénování modelu", style='Menu.TLabel')#, corner_radius= 6, fg_color="#78faae", text_color="#0e3a2f", font=ttk.Font(weight="bold"))
        self.training_label.grid(row=0, column=0, padx=pad/2, pady= (15,20), sticky="nsew")

        self.btFunkce1 = ttk.Button(self.frTrenink, text="Ukládání", command=self.selectTrainpicfolder, compound=tk.LEFT,width=20)
        self.btFunkce1.grid(row=1, column=0, padx=pad, pady=(pad/2,pad/2), sticky='nsew')

        self.btFunkce2 = ttk.Button(self.frTrenink, text="Foto", command=self.capturephoto, compound=tk.LEFT,width=20)
        self.btFunkce2.grid(row=2, column=0, padx=pad, pady=(pad/2,pad/2), sticky='nsew')

        self.btFunkce3 = ttk.Button(self.frTrenink, text="Trénink", compound=tk.LEFT,width=20)
        self.btFunkce3.grid(row=3, column=0, padx=pad, pady=(pad/2,pad/2), sticky='nsew')

        self.evaluation_label = ttk.Label(self.frVyhodnoceni, text="Použití modelu", style='Menu.TLabel')#, corner_radius= 6, fg_color="#78faae", text_color="#0e3a2f", font=ttk.Font(weight="bold"))
        self.evaluation_label.grid(row=0, column=0, padx=pad/2, pady= (15,20), sticky="nsew")

        self.btFunkce4 = ttk.Button(self.frVyhodnoceni, text="Model", command=self.selectmodelpath, compound=tk.LEFT,width=20)
        self.btFunkce4.grid(row=1, column=0, padx=pad, pady=(2,2), sticky='nsew')

        self.btFunkce5 = ttk.Button(self.frVyhodnoceni, text="Ukládat",  command=self.selectshotsfolder, compound=tk.LEFT,width=20)
        self.btFunkce5.grid(row=2, column=0, padx=pad, pady=(2,2), sticky='nsew')

        self.btFunkce6 = ttk.Button(self.frVyhodnoceni, text="Trigger", command=self.start_trigger, compound=tk.LEFT,width=20)
        self.btFunkce6.grid(row=3, column=0, padx=pad, pady=(2,2), sticky='nsew')

        self.btFunkce7 = ttk.Button(self.frVyhodnoceni, text="Trigger", command=self.stop_trigger, compound=tk.LEFT,width=20)
        self.btFunkce7.grid(row=4, column=0, padx=pad, pady=(2,pad), sticky='nsew')

    # Settings window
    def openNastaveni(self):
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
    def load_variables(self, path:str):
        """
        Pokusí se načíst proměnné z cesty path.
        Pokud soubor neexistuje, nastaví vše do defaultu.
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
    def save_variables(self, path:str):
        """ Uloží aktuální nastavení proměnných do souboru path """
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
    def round_and_update_var(self, var):
        # Round the DoubleVar value to one decimal place
        rounded_value = round(var.get(), 1)
        # Set the rounded value back to the DoubleVar
        var.set(rounded_value)

    def set_resolution(self):
        print("start")
        resolution_str=self.Res.get()
        width, height = map(int, resolution_str.split('x'))
        print(width)
        print(height)
        self.camera.change_resolution((width,height))
        print("done")

    def video_stream(self):
        img = self.camera.capture_img()
        
        if img is not None:
            
            image = Image.fromarray(img)
            self.current_img_ref= PIL.ImageTk.PhotoImage(image)
            self.backround_img = self.Imgcanvas.create_image((self.Imgcanvas.winfo_width()/2),(self.Imgcanvas.winfo_height()/2), image=self.current_img_ref)
        img=cv2.resize(img, (self.Imgcanvas.winfo_width(), self.Imgcanvas.winfo_height()))
        self.current_img_ref=PIL.ImageTk.PhotoImage(PIL.Image.fromarray(img))
        self.Imgcanvas.itemconfigure(self.backround_img, image=self.current_img_ref)


        self.video=self.after(10, self.video_stream)

    def selectTrainpicfolder(self):
        self.dataset_path=filedialog.askdirectory()
        print(self.dataset_path)

    def capturephoto(self):
        if self.dataset_path:
            self.btFunkce2.configure(fg_color="#78faae")
            self.camera.capture_img_and_save(filename=datetime.now().strftime("%d.%m.%H.%M") + ".png", folder_path=self.dataset_path)
        else:
            self.btFunkce2.configure(fg_color="red")

    def selectmodelpath(self):
        self.model_path=filedialog.askdirectory()
        print(self.model_path)

    def selectshotsfolder(self):
        self.shots_path=filedialog.askdirectory()
        print(self.shots_path)

    def start_trigger(self):
        #self.tabview.set("Blank")
        self.btNastaveni.configure(state="disabled")
        self.btFunkce1.configure(state="disabled")
        self.btFunkce2.configure(state="disabled")
        self.btFunkce3.configure(state="disabled")
        self.btFunkce4.configure(state="disabled")
        self.btFunkce5.configure(state="disabled")
        self.btFunkce6.configure(state="disabled")
        
    
    def stop_trigger(self):
        #self.tabview.set("Webcam")
        self.btNastaveni.configure(state="normal")
        self.btFunkce1.configure(state="normal")
        self.btFunkce2.configure(state="normal")
        self.btFunkce3.configure(state="normal")
        self.btFunkce4.configure(state="normal")
        self.btFunkce5.configure(state="normal")
        self.btFunkce6.configure(state="normal")

    # Closing routine for saving of variables and termination of window
    def on_closing(self):
        self.save_variables(self.variables_file_path)
        self.camera.release()
        self.destroy()

if __name__ == "__main__":
    # Application start
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
    
