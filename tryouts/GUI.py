#GUI for Raspberry/USB smart cam v.1.6
#Main window + settings window
#20.6. 2024 - Aded tab view, Variables - loading, saving, handling; Auto creation of settings file
#24.6. 2024 - Modified usageoftkinter variables



import customtkinter as ctk
import os
from PIL import Image
from camera_module import Raspicam  # Assuming the camera module code is saved as camera_module.py
from tkinter import filedialog

Nadpis = "ŠKODA SmartCam"
pad = 10

ctk.set_default_color_theme("D:/Python/Programy/Škoda/skoda_project/skodaCI.json")
ctk.set_appearance_mode("dark")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.stop_stream = False
        # Window properties
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        self.geometry("%dx%d" % (width, height))
        self.minsize(400, 300)
        self.title(Nadpis)

        self.settings_icon = ctk.CTkImage(Image.open(os.path.dirname(__file__) + "/settings_icon.png"), size=(26, 26))

        # Initialize tkinter variables
        self.Sour = ctk.StringVar()
        self.Res = ctk.StringVar()
        self.Bri = ctk.DoubleVar()
        self.Con = ctk.DoubleVar()
        self.Fram = ctk.DoubleVar()
        self.Exp = ctk.DoubleVar()
        self.Sat = ctk.DoubleVar()
        self.Sha = ctk.DoubleVar()

        

        # Loading variables at start
        self.variables_file_path = "values.txt"
        self.load_variables()
        self.camera = Raspicam()
        self.nactiGUI()
        self.video_stream()
        
        

    # GUI init
    def nactiGUI(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Frames
        self.frOvladani = ctk.CTkFrame(self)
        self.frOvladani.grid(row=0, column=0, padx=pad, pady=pad, sticky="nsew")

        self.frObraz = ctk.CTkFrame(self)
        self.frObraz.grid(row=0, column=1, columnspan=2, padx=pad, pady=pad, sticky='nsew')
        self.frObraz.grid_columnconfigure(0, weight=1)
        self.frObraz.grid_rowconfigure(0, weight=1)

        # Tabview
        self.tabview = ctk.CTkTabview(self.frObraz)
        self.tabview.grid(row=0, column=0, padx=pad, pady=(pad/2), sticky="nsew")
        self.tabview.add("Webcam")
        self.tabview.add("Blank")

        self.tabview.tab("Blank").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Blank").grid_rowconfigure(0, weight=1)

        # Widgets- Right frame - displaying values from settings
        self.lA = ctk.CTkLabel(self.tabview.tab("Webcam"), textvariable=self.Sour)
        self.lA.grid(row=0, column=0, padx=pad, pady=pad, sticky='nsew')

        self.lB = ctk.CTkLabel(self.tabview.tab("Webcam"), textvariable=self.Res)
        self.lB.grid(row=1, column=0, padx=pad, pady=pad, sticky='nsew')

        self.lC = ctk.CTkLabel(self.tabview.tab("Webcam"), textvariable=self.Bri)
        self.lC.grid(row=2, column=0, padx=pad, pady=pad, sticky='nsew')

        self.lD = ctk.CTkLabel(self.tabview.tab("Webcam"), textvariable=self.Con)
        self.lD.grid(row=3, column=0, padx=pad, pady=pad, sticky='nsew')

        self.lE = ctk.CTkLabel(self.tabview.tab("Webcam"), textvariable=self.Fram)
        self.lE.grid(row=4, column=0, padx=pad, pady=pad, sticky='nsew')

        self.lF = ctk.CTkLabel(self.tabview.tab("Webcam"), textvariable=self.Exp)
        self.lF.grid(row=5, column=0, padx=pad, pady=pad, sticky='nsew')

        self.lG = ctk.CTkLabel(self.tabview.tab("Webcam"),textvariable=self.Sat)
        self.lG.grid(row=6, column=0, padx=pad, pady=pad, sticky='nsew')

        self.lH = ctk.CTkLabel(self.tabview.tab("Webcam"), textvariable=self.Sha)
        self.lH.grid(row=7, column=0, padx=pad, pady=pad, sticky='nsew')

        self.video_label = ctk.CTkLabel(self.tabview.tab("Blank"), text="")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Widgets - Left frame
        self.btNastaveni = ctk.CTkButton(self.frOvladani, text="Nastavení", height=30, width=30, anchor='center', image=self.settings_icon, command=self.openTopLevel)
        self.btNastaveni.grid(row=0, column=0, padx=pad, pady=pad, sticky='e')

        self.btFunkce1 = ctk.CTkButton(self.frOvladani, text="Trenink sl.", height=30, width=30, anchor='center', command=self.selectTrainpicfolder)
        self.btFunkce1.grid(row=1, column=0, padx=pad, pady=pad, sticky='nsew')

        self.btFunkce2 = ctk.CTkButton(self.frOvladani, text="Trénink start", height=30, width=30, anchor='center')
        self.btFunkce2.grid(row=2, column=0, padx=pad, pady=pad, sticky='nsew')

        self.btFunkce3 = ctk.CTkButton(self.frOvladani, text="Model", height=30, width=30, anchor='center', command=self.selectmodelpath)
        self.btFunkce3.grid(row=3, column=0, padx=pad, pady=pad, sticky='nsew')

        self.btFunkce4 = ctk.CTkButton(self.frOvladani, text="Snímky", height=30, width=30, anchor='center', command=self.selectshotsfolder)
        self.btFunkce4.grid(row=5, column=0, padx=pad, pady=pad, sticky='nsew')

    # Settings window
    def openTopLevel(self):
        self.topLevel = ctk.CTkToplevel(self,)
        self.topLevel.title("ŠKODA SmartCam - Nastavení")
        self.topLevel.resizable(False, False)

        # Nadpis
        ctk.CTkLabel(self.topLevel, text="Nastavení kamery", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=pad, pady=pad, columnspan=3)

        # Settings Item
        # Source
        ctk.CTkLabel(self.topLevel, text="Zdroj:", anchor='w').grid(row=1, column=0, padx=pad, pady=pad)

        ctk.CTkOptionMenu(self.topLevel, values=["RasPi", "USB"], variable=self.Sour).grid(row=1, column=1, padx=pad, pady=pad)

        # Resolution
        ctk.CTkLabel(self.topLevel, text="Rozlišení:", anchor='w').grid(row=2, column=0, padx=pad, pady=pad)

        ctk.CTkOptionMenu(self.topLevel, values=["1920x1080","1280x720", "640x480"], variable=self.Res).grid(row=2, column=1, padx=pad, pady=pad)

        # Brightness
        ctk.CTkLabel(self.topLevel, text="Jas:", anchor='w').grid(row=3, column=0, padx=pad, pady=pad)

        ctk.CTkSlider(self.topLevel, from_=30, to=255, variable=self.Bri, number_of_steps=15).grid(row=3, column=1, padx=pad, pady=pad)
        ctk.CTkLabel(self.topLevel, textvariable=self.Bri, anchor='w').grid(row=3, column=3, padx=pad, pady=pad)

        # Contrast
        ctk.CTkLabel(self.topLevel, text="Kontrast:", anchor='w').grid(row=4, column=0, padx=pad, pady=pad)

        ctk.CTkSlider(self.topLevel, from_=0, to=10, variable=self.Con, number_of_steps=10).grid(row=4, column=1, padx=pad, pady=pad)
        ctk.CTkLabel(self.topLevel, textvariable=self.Con, anchor='w').grid(row=4, column=3, padx=pad, pady=pad)

        # Framerate
        ctk.CTkLabel(self.topLevel, text="Framerate:", anchor='w').grid(row=5, column=0, padx=pad, pady=pad)

        ctk.CTkSlider(self.topLevel, from_=-0.8, to=0.8, variable=self.Fram, number_of_steps=16).grid(row=5, column=1, padx=pad, pady=pad)
        ctk.CTkLabel(self.topLevel, textvariable=self.Fram, anchor='w').grid(row=5, column=3, padx=pad, pady=pad)

        # Exposure
        ctk.CTkLabel(self.topLevel, text="Expozice:", anchor='w').grid(row=6, column=0, padx=pad, pady=pad)

        ctk.CTkSlider(self.topLevel, from_=0, to=100, variable=self.Exp, number_of_steps=10).grid(row=6, column=1, padx=pad, pady=pad)
        ctk.CTkLabel(self.topLevel, textvariable=self.Exp, anchor='w').grid(row=6, column=3, padx=pad, pady=pad)

        # Saturation
        ctk.CTkLabel(self.topLevel, text="Saturace:", anchor='w').grid(row=7, column=0, padx=pad, pady=pad)

        ctk.CTkSlider(self.topLevel, from_=0, to=200, variable=self.Sat, number_of_steps=10).grid(row=7, column=1, padx=pad, pady=pad)
        ctk.CTkLabel(self.topLevel, textvariable=self.Sat, anchor='w').grid(row=7, column=3, padx=pad, pady=pad)

        # Sharpness
        ctk.CTkLabel(self.topLevel, text="Ostrost:", anchor='w').grid(row=8, column=0, padx=pad, pady=pad)

        ctk.CTkSlider(self.topLevel, from_=0, to=50, variable=self.Sha, number_of_steps=10).grid(row=8, column=1, padx=pad, pady=pad)
        ctk.CTkLabel(self.topLevel, textvariable=self.Sha, anchor='w').grid(row=8, column=3, padx=pad, pady=pad)

        #Definition of CTk Doublevar tracing - for rounding them 
        self.topLevel.attributes('-topmost', 'true')
        self.Bri.trace_add("write", lambda *args: (self.round_and_update_var(self.Bri), self.camera.set_controls(brightness=self.Bri.get())))
        self.Con.trace_add("write", lambda *args: (self.round_and_update_var(self.Con), self.camera.set_controls(contrast=self.Con.get())))
        self.Fram.trace_add("write", lambda *args: self.round_and_update_var(self.Fram))
        self.Exp.trace_add("write", lambda *args: self.round_and_update_var(self.Exp))
        self.Sat.trace_add("write", lambda *args: (self.round_and_update_var(self.Sat), self.camera.set_controls(saturation=self.Sat.get())))
        self.Sha.trace_add("write", lambda *args: (self.round_and_update_var(self.Sha), self.camera.set_controls(sharpness=self.Sha.get())))
        
        self.Fram.trace_add("write", lambda *args: self.set_resolution())

    # Loading of variables values
    def load_variables(self):
        default_values = {"Sour": "USB", "Res": "640x480", "Bri": 1, "Con": 6, "Fram": 0.8, "Exp": 0.8, "Sat": 0.8, "Sha": 0.8}

        if os.path.exists(self.variables_file_path):
            with open(self.variables_file_path, 'r') as file:
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
            self.save_variables()


    # Saving variables values
    def save_variables(self):
        with open(self.variables_file_path, 'w') as file:
            file.write(f"Sour={self.Sour.get()}\n")
            file.write(f"Res={self.Res.get()}\n")
            file.write(f"Bri={self.Bri.get()}\n")
            file.write(f"Con={self.Con.get()}\n")
            file.write(f"Fram={self.Fram.get()}\n")
            file.write(f"Exp={self.Exp.get()}\n")
            file.write(f"Sat={self.Sat.get()}\n")
            file.write(f"Sha={self.Sha.get()}\n")

    #Funtion for rounding of CTk DoubleVar
    def round_and_update_var(self, var):
        # Round the DoubleVar value to one decimal place
        rounded_value = round(var.get(), 1)
        # Set the rounded value back to the DoubleVar
        var.set(rounded_value)

    def set_resolution(self,*args):
        print("start")
        resolution_str=self.Res.get()
        width, height = map(int, resolution_str.split('x'))
        self.camera.change_resolution(width,height)
        print("done")

    def video_stream(self):
        img = self.camera.capture_img()
        if img is not None:
            image = Image.fromarray(img)
            ctk_image = ctk.CTkImage(light_image=image, dark_image=image,size=(self.tabview.tab("Blank").winfo_width(),self.tabview.tab("Blank").winfo_height()))
            self.video_label.image = ctk_image
            self.video_label.configure(image=ctk_image)

    def selectTrainpicfolder(self):
        self.Lpicture_path=filedialog.askdirectory()
        print(self.Lpicture_path)

    def selectmodelpath(self):
        self.model_path=filedialog.askdirectory()
        print(self.model_path)

    def selectshotsfolder(self):
        self.shots_path=filedialog.askdirectory()
        print(self.shots_path)

    # Closing routine for saving of variables and termination of window
    def on_closing(self):
        self.save_variables()
        self.camera.release()
        self.stop_stream = True
        self.destroy()

if __name__ == "__main__":
    # Application start
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
