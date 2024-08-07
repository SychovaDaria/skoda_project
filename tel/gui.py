import os
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog, PhotoImage, messagebox
import numpy as np
import cv2
from datetime import datetime
from training_module import ModelTrainer
from trigger_module import PhoneDetector
from raspicam import Raspicam
import io
import base64

Title = "ŠKODA SmartCam"
pad = 10

ctk.set_appearance_mode("dark")


# HEX codes for icons
icon_setting ="iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAPESURBVEhLrZbLb1tVEIdnfOtHEjtOilCI1PBII1Q1NKGNSJWUypvSsqHioUpI/AUINlSoLJDYIB5iAxvWqEIqi4iHImCBgCoQnFQlkLqoKpIXqI1l0pKHfZPe2Pje4XeOT64TP24t1E+yz5k54zPnzMyda6Y26Tx+pN+quB9i+hyxuCz0uet5r29e+mO5ahFMe44OHowkesLzMD5sNBoRuWhHek/SzEzFqFoSMmMg3cnIKG4wYkQfJp5MOmsDRgykHUe4iLcf35aRazBFyZJBIwWyM3QcT43dZ21V4m6YNzdmF//pmhgZxu4fYOkULFsdShDDabHonD2byXYdfex+i8MJcZ01+/L1FWNTcxSfHD0H4Q189iL261i5iC1OMHPCmASC3xQQyu+F5SmM3VDcFuF37fkrH6l17Sg+MfpsiOmLbfmeIeKKFzptX1r8VoeDSV7Ww72G2SLLe0lNjSPuUONdECG6hRB9h6T8gPmq0lWXWsPCMTVWE8zymR5bIVSCzZsR2hqy5zKn7HTmRLgSfUSY3ob7oGcI5/Km1USXbGnwwGLEc4Zws2GI9SEUCslbxV8y7zs3V0pGR04uVyrfXJ6JDvR1IUTHjLoGPGCn8/a+A+/QtWs40zZnzliJpetp5tC40VRB9YS5tH8l/adtNLvoHRtLViKVHDN1GZXG8+jrjfkrpzHV4a09G1NTLpx0G8kHVr+3cqJYW1goIKy/GtGHWeIY/BzWPYRNctusI9QjXLePZtdmNQOEDkvrRtoBH947Pt5w022STx7qRfyfMKIPqmATg5+a6mlTqVhiPXcBXeCklncAy5hruf+qxBtVDRwuWlx9D0YNxYC9Ho082DdUHk58Q9lVVztK9Pe8imSe1RaNINxyTFVXZ9++y1v5vK48dRPtROgVOGoWOqj5UNSJ5EpLywv6at2TI7NQN5ZoHXhiN3XikRMdLib9MAYjXxbTmee3T9IkN42oEsYpUxiPt+dEwRvqWztC6/sYx73rW/J/UGbxzquJzlE5dysbG3hgGS3+iG7xRH+jn11AiAZx+nb6oAJ9kD5Fg34IV+/E72/g1meL6atfqUW//DT4b9Df0bEnXyi4lM2WkkdHH5aQqPcJXnzV5tiIOHjvTLt7rNfu/Pxbnp4eivbfTlp5x6mg9ZSNUZ2j5nBy4vEXcNspI+/CEy+1MXf1JyO2pGlZ1oEDV/7Ct2tkH4Rqy/JCN4wYSDuOqLjuZXCxOSP6IB8/FmI9S0YMpC1HKtYcoWeqBSIljHdwn0/CHH+xnf90RET/AaT0aaZ75bKqAAAAAElFTkSuQmCC"
icon_start ="iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAI9SURBVEhLtZbBa9NwFMffSxoYSlqmAw9OxKNCowM77aj0IHgVRVEQUfTkQfDizYP/gKJH8SCM7bSjguBJsaazUlw7C4J6EbUw5tR0UxeXPN8vfch+rrOZJB9o8t63aT/8fgn5/RB6kDvk7KIArxDAcb5gmAi+ANITArq56M5W+BL+amOsEdkl5wAG8AAQhyRajU9Etzq+dQ3q9V+SxUIT5Ur5wTDEWQTcLlFvCCqAK+c9t/VOkr4Yco4IA7zYV6JAKBFlXmSLzumoi4Em4ntxTMq+IMIgHyfs4t7bW0ZHsxKviyYyELdKGQ8Ek4WXVzI/n24+mM9L2hNN9P+gYxg4bRedC1AuZyTUSEikBoebEPGu7X+dsAuFNTOTmEjg2YdTYC3Xs2Mj+yWLSFoUwaPbSRQ+yhWdEYnSESnUUxkijsNJMFWfmkiBRLvtT/mCqlMV8bBMCHGPKtMVKRBDdUpXRLwGELZUmaqIABtedaau6tREvJzMEwRnuExx6ohaoWkeXqy+ei1JwiK+J7wK38kEA2NLlZdNSSMSFNFnIDzXcZuXFmo1T8I/JCMiqlJgFL3pxqTquqHOXyKalyIuPn9ueL5V7jyfedONeqOJeH6npOwPUZuQTnhu42qcjYomsoIf9/gfPkq7Hvzk0uMAYF/nWfO+6rvxv9FEC7W3Hhp0lH/5QSINHvF3Pl7n7daRpWpzTuJYRK/w1Sy/n2tbw9smeU0ZQKAd/K6yWeDxVuehicbZb25jCtptHtBGAPgNmAvFtYY+MBcAAAAASUVORK5CYII="
icon_stop ="iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAEJSURBVEhL7ZYxSoNBEIXfm98kTbQ0iMQLSKwVreIpvEEaQbC3F7Sw9hKCjQQsbFQsBPUIimgqMSmMmB1n4h5A1v0b+b9md17zMbvNIyLN9ZUNKnah6JKci3ESqvpmR5/Kw/fruxvPpqLZ1c4ORfbtWvc5G4pxYNgeXT4c0zcR5bnFeSURVYwCdK1otFtHBJdjnh0SdXu2BcFX6MasPCZhUzhT/Onjf4M7JN5LpxIlU4mSqUTJVKJk/qEoFolScYdv1P8ZS4Q8E0IOrK18xCg7ts0QE+4V46eX59rS/CuU1ufyNiGXULQ3vLq/KDz4fBzc1tqtU28rNi5agWx4nsr038kTBNlyCQB8A7zhUZfK67ZBAAAAAElFTkSuQmCC"
icon_folder ="iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAKrSURBVEhLtZbPTxNBFMffW9oCjbSgB8JFTiZKsEVrlAjaCwe9EdR48mC86P+giZLe5G6iHtR41AQNKheCjaapGIyp2sQfoCaAGgyGdktJ2c7zu7sTQ6FB6eInm50339nud3be29kyraItFgsW/MUGO2ZfnSzWbc9TMmk5gx5xjMLxaLNakQQLDRBzwBlx+UYkiZy/5b5XQw4e2d/mK1tJhLu0VgX1FpMYDATKT+aTWVOLm4KbeqLXcJPzur8RgmOOhNcbsTM2pURumaXAA5qcXNEjf+DQ4chPNDt03ysCxlis07l0dkFrDsYWmtgw6BP23aNYzK81BxhtPUwcb6ovHdddh/9iBCcDx0ndc0COokjkBggtE8tnRNO4sOSK/8S7fCpzSccbGIlYWPHbQupqPvXmg624A7VR3QgmInQun87ctXutkUiwEDLalVBFgv8Gqt4q/Fr5SNlsqZoR+nI5l8oMUnd3YxMvXcRPLjBTix7fHELvVZkHqhjJrM9q6FhonyiEZiM3YHJWD9QMVufVuqqD+HhhYiLXPNfVidmc0bInsKHuq1beD+0T8tGPYvA5ikewZF8qjPA0ZliCYwgxCTnhqt5BfsfXPJGkZtLp5fChaDtc92jRM0rRSIURnIfRiGI5unXLJktU73u2ykiKbNEoAvjxKVfzDu41bZq0uMqIvy6W/TPh3r3N2HIOaNEzyPsL+/tkIJrXwlNbUEp245lanau8I0zKLi7s3kwjjkTyyG3qjuFc8aeldqRoBRrG7chQhjGEOv+hyxpIv9t6R1iGlpIvv9uxYT5/nWVl9KKsiy09HTuRvE7nKi9gy0ft3skvBxJaqVyiUHf0IGZxHfI2LW0K3KwMlyll0E2zLzNMV2CoWZ+LeNx+f2p7hxpnhUY/2R/HNRs10W9QMgoy4ZiUXgAAAABJRU5ErkJggg=="
icon_model="iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAJ2SURBVEhL5ZbPaxNBFMffzKalYJKmCApFe1EPpjENCIZtxFwVPBXrD7wJ/gcehFIEb7156kFQ8FCPetFeFD1oEwmCJEIhBUVaa7FaKdmKsUn3+Wby2Gya3bBuvfkhk3nvuzP7zbyd3Y2AXUSzo8elEblB4XlqB7S4i1qhrOdFc+mJ/sbA8x+lUk0f6IHkXhPLnjgnZKRE4TVqniZuhA13mpH620QuM8aSL47RUO7kCEg5JwREWQoCrUwc27HxZWw8ZbLmiWPUxMZ1Mhni9K/Q81DOx8zMOEtdtEuHcJajUAghElTLp35m7mt0kPvQCCAzwCdeZm6jrh0YBl1GYc9HT2WSLGncRsFJJvs52ua+A1rZoDTsu5DPR1gKZzSY6NO/lkowhYCPPZuA9YHm1rCeQDjlipnpFbqghzjtiTqRtdm8DIuLnivyIpQRgfR5hSBnDAGrfm4G7tStwvuqisMaBQIRP1rFyhEVh9sMIfjfjBCWUNimLcVoz4Z2nnbHZ57lSc/NgAiPrGL5ggpbig+Tk0ZstfqCbtQzrGjcm6H3rkOwacRrCixW/KCnvlCvCed8iuBGeySwEZXOEgLvI4pNltyk6AE6QX3HKtz4GS2T0WFONTTwIQ28ymknt0DGn6XX6RT7WemC5n+g+UdV7BjFzbE3lGU5ZXCDvqbRhp+tvA0akJIo1J8Y3xWR00KtWDmtQpdReppeJrc5/Tcg3KwVyzMqdO6jhjDu0S/4zumeobIt94n6LKdto1+Fd1/oGl2kEd9YCg/iGki8tFGoOreFwb3m98rXT3JkeM5AO05lVBtjX+tIMGgVa9Q9QENe2VqoLLVUBcAfL9b18x24LDwAAAAASUVORK5CYII="
icon_camera="iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAK1SURBVEhLvZZNaBNREIBnXnY3YrKbpApFKv5cRBB3K8FDU6QnT54UtIJeepAKUvDYilisB7FIQahgPXkQr148C8ES8VDaFPVc0ZOmrUnaotnNPudtXtKYbNpNXfxg2Zl5j5md9+bNW+w1zdhWDO8Awk0E6IEWOPDX5dzyJan6YmTMNwB4QarbcF4g+5NS0Z5mIggiTPgF+WcQD1ICU0ZSmUA9Y62KIJzzWRdgDiOMXtuoaJfX3336KlVfUoPpIzZW41L14E41whjeQsBRDrCGRsaiN0CV89Ob75c/erNCIpGx+sn5opCZZyFaMwkDG9FLQtDIyGV4amN+6bNn9UE/e/IAKNGnNFOrWVhFqSij6wsLxZrezv7BfkvhfEnIjYx25DJEQI3OUtEMI7KLtQeGbdWZodFAPnaclEqnE5TxNeOb+ZgW4Yo0N2AI1/UB64ExYF7tPW/GpNkXJs6JeER1SZtHYshKOlFnnsSXVKa3qUzbPwpBo8zoDOKrrU14C0NH98kRD9Xhxbp/qmx/4hnrBnl+LtVA0GaPlHP5F1L9i45LR0vVJ8Vm6OPotHNY9eRWOByTUhuBNlLCXc4nS1rqUEn7eZi77iNpD0TwQJyvbmiph5DNOpD98qtsa/copZIc3ZUuMqJthxVFKrSJhQgtVcc9biV4IIQe/bcxCem0KqorXjGmKbQuR3elmz2ic8rG9ahT0CvJ7wzZmLQHoptAHrRWRjeZ1GHi0hKPaPXSFhqJc+bxuv+OTVW0eJfDGakGglrSYjGX95qooLmpBu7ee6H77h0C/z+QuOOlGBqq6zb8I90lP8TfCnXLOQb4rPn6FYhWX/yQX5GqL6K6bJcZUvUQQaoIY+RzRDRiCmTdpcMxRWO+7UTcJXv+r6vhkuNxJn7uyN19crgmB8JDZEJBin0nZv4ADD/yitiKF+wAAAAASUVORK5CYII="
skoda_logo="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"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window properties
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        self.geometry("%dx%d" % (width, height))
        self.minsize(400, 300)
        self.title("ŠKODA SmartCam")
        self.Tpicture_path = ""

        # Load icons
        self.icon_setting = ctk.CTkImage(light_image=Image.open(io.BytesIO(base64.b64decode(icon_setting))), dark_image=Image.open(io.BytesIO(base64.b64decode(icon_setting))), size=(20, 20))
        self.icon_start = ctk.CTkImage(light_image=Image.open(io.BytesIO(base64.b64decode(icon_start))), dark_image=Image.open(io.BytesIO(base64.b64decode(icon_start))), size=(20, 20))
        self.icon_stop = ctk.CTkImage(light_image=Image.open(io.BytesIO(base64.b64decode(icon_stop))), dark_image=Image.open(io.BytesIO(base64.b64decode(icon_stop))), size=(20, 20))
        self.icon_folder = ctk.CTkImage(light_image=Image.open(io.BytesIO(base64.b64decode(icon_folder))), dark_image=Image.open(io.BytesIO(base64.b64decode(icon_folder))), size=(20, 20))
        self.icon_model = ctk.CTkImage(light_image=Image.open(io.BytesIO(base64.b64decode(icon_model))), dark_image=Image.open(io.BytesIO(base64.b64decode(icon_model))), size=(20, 20))
        self.icon_camera = ctk.CTkImage(light_image=Image.open(io.BytesIO(base64.b64decode(icon_camera))), dark_image=Image.open(io.BytesIO(base64.b64decode(icon_camera))), size=(20, 20))
        self.skoda_logo = ctk.CTkImage(light_image=Image.open(io.BytesIO(base64.b64decode(skoda_logo))), dark_image=Image.open(io.BytesIO(base64.b64decode(skoda_logo))), size=(80, 80))

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
        self.camera = Raspicam(use_usb=False)
        self.initGUI()

        self.video_thread = threading.Thread(target=self.video_stream, daemon=True)
        self.video_thread.start()

        # ROI variables
        self.roi = []
        self.roi_bool = False

    # GUI initialization
    def initGUI(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Frames
        self.control_frame = ctk.CTkFrame(self, width=227)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.training_frame = ctk.CTkFrame(self.control_frame, fg_color="#264e44")
        self.training_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.evaluation_frame = ctk.CTkFrame(self.control_frame, fg_color="#264e44")
        self.evaluation_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.grid(row=0, column=1, columnspan=2, padx=10, pady=10, sticky='nsew')
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        # Tabview
        self.tabview = ctk.CTkTabview(self.image_frame, state="disabled")
        self.tabview.grid(row=0, column=0, padx=10, pady=(5), sticky="nsew")
        self.tabview.add("Webcam")
        self.tabview.add("Blank")

        self.tabview.tab("Webcam").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Webcam").grid_rowconfigure(0, weight=1)

        self.tabview.tab("Blank").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Blank").grid_rowconfigure(0, weight=1)

        # Video labels
        self.video_label_webcam = ctk.CTkLabel(master=self.tabview.tab("Webcam"), text="")
        self.video_label_webcam.grid(row=0, column=0, sticky="nsew")

        self.video_label_blank = ctk.CTkLabel(master=self.tabview.tab("Blank"), text="")
        self.video_label_blank.grid(row=0, column=0, sticky="nsew")

        self.video_label_webcam.bind("<Button-1>", self.on_click)
        self.video_label_webcam.bind("<ButtonRelease-1>", self.on_release)
        self.video_label_webcam.bind("<B1-Motion>", self.on_motion)

        # Left frame widgets
        self.logo_label = ctk.CTkLabel(self.control_frame, text="", fg_color="#0e3a2f", image=self.skoda_logo)
        self.logo_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.btNastaveni = ctk.CTkButton(self.control_frame, text="Camera Settings", height=30, anchor='center', image=self.icon_setting, command=self.openTopLevel)
        self.btNastaveni.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.training_label = ctk.CTkLabel(self.training_frame, text="Train Model", corner_radius=6, fg_color="#78faae", text_color="#0e3a2f", font=ctk.CTkFont(weight="bold"))
        self.training_label.grid(row=0, column=0, padx=5, pady=(15, 20), sticky="nsew")

        self.btFunkce2 = ctk.CTkButton(self.training_frame, text="Capture Photo", height=30, anchor='center', image=self.icon_camera, command=self.capturephoto)
        self.btFunkce2.grid(row=1, column=0, padx=10, pady=(5, 5), sticky='nsew')

        self.btFunkce3 = ctk.CTkButton(self.training_frame, text="Train", height=30, anchor='center', image=self.icon_start, command=self.start_training)
        self.btFunkce3.grid(row=2, column=0, padx=10, pady=(5, 5), sticky='nsew')

        self.evaluation_label = ctk.CTkLabel(self.evaluation_frame, text="Use Model", corner_radius=6, fg_color="#78faae", text_color="#0e3a2f", font=ctk.CTkFont(weight="bold"))
        self.evaluation_label.grid(row=0, column=0, padx=5, pady=(15, 20), sticky="nsew")

        self.btFunkce4 = ctk.CTkButton(self.evaluation_frame, text="Select Model", height=30, anchor='center', image=self.icon_model, command=self.selectmodelpath)
        self.btFunkce4.grid(row=1, column=0, padx=10, pady=(2, 2), sticky='nsew')

        self.btFunkce5 = ctk.CTkButton(self.evaluation_frame, text="Select Folder", height=30, anchor='center', image=self.icon_folder, command=self.selectshotsfolder)
        self.btFunkce5.grid(row=2, column=0, padx=10, pady=(2, 2), sticky='nsew')

        self.btFunkce6 = ctk.CTkButton(self.evaluation_frame, text="Trigger", height=30, image=self.icon_start, anchor='center', command=self.start_trigger)
        self.btFunkce6.grid(row=3, column=0, padx=10, pady=(2, 2), sticky='nsew')

        self.btFunkce7 = ctk.CTkButton(self.evaluation_frame, text="Stop Trigger", height=30, image=self.icon_stop, anchor='center', command=self.stop_trigger)
        self.btFunkce7.grid(row=4, column=0, padx=10, pady=(2, 10), sticky='nsew')

        self.btRoi = ctk.CTkButton(self.evaluation_frame, text="ROI", height=30, image=self.icon_stop, anchor='center', command=self.start_roi_selection)
        self.btRoi.grid(row=5, column=0, padx=10, pady=(2, 10), sticky='nsew')

    def on_click(self, event):
        if self.roi_bool:
            self.roi.append(np.floor(np.array([event.x, event.y]) / np.array([self.video_label_webcam.winfo_width(), self.video_label_webcam.winfo_height()]) * self.camera.resolution))

    def on_release(self, event):
        if self.roi_bool:
            self.roi[-1] = list(self.roi[-1]) + list(np.floor(np.array([event.x, event.y]) / np.array([self.video_label_webcam.winfo_width(), self.video_label_webcam.winfo_height()]) * self.camera.resolution))
            for i in range(len(self.roi[-1])):
                self.roi[-1][i] = int(self.roi[-1][i])
            self.roi_bool = False
            print(self.roi)

    def on_motion(self, event):
        pass

    def start_roi_selection(self):
        self.roi_bool = True
        print("ROI selection started")

    # Settings window
    def openTopLevel(self):
        self.topLevel = ctk.CTkToplevel(self)
        self.topLevel.title("ŠKODA SmartCam - Settings")
        self.topLevel.resizable(False, False)

        # Title
        ctk.CTkLabel(self.topLevel, text="Camera Settings", font=ctk.CTkFont(size=20, weight="bold"), fg_color="#264e44", text_color="#78faae", anchor="center").grid(row=0, column=0, padx=10, pady=10, columnspan=3, sticky="nsew")

        # Settings Item
        # Source
        ctk.CTkLabel(self.topLevel, text="Source:", font=ctk.CTkFont(weight="bold"), anchor='w').grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        ctk.CTkOptionMenu(self.topLevel, values=["RasPi", "USB"], variable=self.Sour).grid(row=1, column=1, padx=10, pady=10)

        # Resolution
        ctk.CTkLabel(self.topLevel, text="Resolution:", font=ctk.CTkFont(weight="bold"), anchor='w').grid(row=2, column=0, padx=10, pady=10, sticky='nsew')
        ctk.CTkOptionMenu(self.topLevel, values=["1920x1080", "1280x720", "640x480"], variable=self.Res).grid(row=2, column=1, padx=10, pady=10)

        # Brightness
        ctk.CTkLabel(self.topLevel, text="Brightness:", font=ctk.CTkFont(weight="bold"), anchor='w').grid(row=3, column=0, padx=10, pady=10, sticky='nsew')
        ctk.CTkSlider(self.topLevel, from_=30, to=255, variable=self.Bri, number_of_steps=15).grid(row=3, column=1, padx=10, pady=10)
        ctk.CTkLabel(self.topLevel, textvariable=self.Bri, anchor='w', fg_color="#0e3a2f", text_color="#78faae").grid(row=3, column=2, padx=10, pady=10, sticky='nsew')

        # Contrast
        ctk.CTkLabel(self.topLevel, text="Contrast:", font=ctk.CTkFont(weight="bold"), anchor='w').grid(row=4, column=0, padx=10, pady=10, sticky='nsew')
        ctk.CTkSlider(self.topLevel, from_=0, to=10, variable=self.Con, number_of_steps=10).grid(row=4, column=1, padx=10, pady=10)
        ctk.CTkLabel(self.topLevel, textvariable=self.Con, anchor='w', fg_color="#0e3a2f", text_color="#78faae").grid(row=4, column=2, padx=10, pady=10, sticky='nsew')

        # Framerate
        ctk.CTkLabel(self.topLevel, text="Framerate:", font=ctk.CTkFont(weight="bold"), anchor='w').grid(row=5, column=0, padx=10, pady=10, sticky='nsew')
        ctk.CTkSlider(self.topLevel, from_=0, to=30, variable=self.Fram, number_of_steps=16).grid(row=5, column=1, padx=10, pady=10)
        ctk.CTkLabel(self.topLevel, textvariable=self.Fram, anchor='w', fg_color="#0e3a2f", text_color="#78faae").grid(row=5, column=2, padx=10, pady=10, sticky='nsew')

        # Exposure
        ctk.CTkLabel(self.topLevel, text="Exposure:", font=ctk.CTkFont(weight="bold"), anchor='w').grid(row=6, column=0, padx=10, pady=10, sticky='nsew')
        ctk.CTkSlider(self.topLevel, from_=0, to=100, variable=self.Exp, number_of_steps=10).grid(row=6, column=1, padx=10, pady=10)
        ctk.CTkLabel(self.topLevel, textvariable=self.Exp, anchor='w', fg_color="#0e3a2f", text_color="#78faae").grid(row=6, column=2, padx=10, pady=10, sticky='nsew')

        # Saturation
        ctk.CTkLabel(self.topLevel, text="Saturation:", font=ctk.CTkFont(weight="bold"), anchor='w').grid(row=7, column=0, padx=10, pady=10, sticky='nsew')
        ctk.CTkSlider(self.topLevel, from_=0, to=200, variable=self.Sat, number_of_steps=10).grid(row=7, column=1, padx=10, pady=10)
        ctk.CTkLabel(self.topLevel, textvariable=self.Sat, anchor='w', fg_color="#0e3a2f", text_color="#78faae").grid(row=7, column=2, padx=10, pady=10, sticky='nsew')

        # Sharpness
        ctk.CTkLabel(self.topLevel, text="Sharpness:", font=ctk.CTkFont(weight="bold"), anchor='w').grid(row=8, column=0, padx=10, pady=10, sticky='nsew')
        ctk.CTkSlider(self.topLevel, from_=0, to=50, variable=self.Sha, number_of_steps=10).grid(row=8, column=1, padx=10, pady=10)
        ctk.CTkLabel(self.topLevel, textvariable=self.Sha, anchor='w', fg_color="#0e3a2f", text_color="#78faae").grid(row=8, column=2, padx=10, pady=10, sticky='nsew')

        # Set controls
        self.topLevel.attributes('-topmost', 'true')
        self.Bri.trace_add("write", lambda *args: (self.round_and_update_var(self.Bri), self.camera.set_controls(brightness=self.Bri.get())))
        self.Con.trace_add("write", lambda *args: (self.round_and_update_var(self.Con), self.camera.set_controls(contrast=self.Con.get())))
        self.Fram.trace_add("write", lambda *args: self.round_and_update_var(self.Fram))
        self.Exp.trace_add("write", lambda *args: self.round_and_update_var(self.Exp))
        self.Sat.trace_add("write", lambda *args: (self.round_and_update_var(self.Sat), self.camera.set_controls(saturation=self.Sat.get())))
        self.Sha.trace_add("write", lambda *args: (self.round_and_update_var(self.Sha), self.camera.set_controls(sharpness=self.Sha.get())))

        self.Res.trace_add("write", lambda *args: self.set_resolution())

    # Load variables values
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

    # Save variables values
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

    # Round CTk DoubleVar
    def round_and_update_var(self, var):
        rounded_value = round(var.get(), 1)
        var.set(rounded_value)

    def set_resolution(self):
        resolution_str = self.Res.get()
        width, height = map(int, resolution_str.split('x'))
        self.camera.change_resolution((width, height))

    def video_stream(self):
        img = self.camera.capture_img()
        for roi in self.roi:
            if len(roi) == 4:
                cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
            if self.tabview.get() == "Webcam":
                ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(self.tabview.tab("Webcam").winfo_width(), self.tabview.tab("Webcam").winfo_height()))
                self.video_label_webcam.image = ctk_image
                self.video_label_webcam.configure(image=ctk_image)
            else:
                ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(self.tabview.tab("Blank").winfo_width(), self.tabview.tab("Blank").winfo_height()))
                self.video_label_blank.image = ctk_image
                self.video_label_blank.configure(image=ctk_image)
        self.after(20, self.video_stream)

    def capturephoto(self):
        if self.Tpicture_path:
            self.btFunkce2.configure(fg_color="#78faae")
            self.camera.capture_img_and_save(filename=datetime.now().strftime("%d.%m.%H.%M") + ".png", folder_path=self.Tpicture_path)
        else:
            self.btFunkce2.configure(fg_color="red")

    def selectmodelpath(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
        print(self.model_path)

    def selectshotsfolder(self):
        self.shots_path = filedialog.askdirectory()
        print(self.shots_path)

    def start_training(self):
        object_folder = filedialog.askdirectory(title="Select Object Folder")
        non_object_folder = filedialog.askdirectory(title="Select Non-Object Folder")
        if object_folder and non_object_folder:
            threading.Thread(target=self.run_training, args=(object_folder, non_object_folder), daemon=True).start()

    def run_training(self, object_folder, non_object_folder):
        try:
            self.trainer = ModelTrainer(object_folder=object_folder, non_object_folder=non_object_folder)
            self.trainer.train()
            messagebox.showinfo("Training", "Model training has started.")
        except ValueError as e:
            messagebox.showerror("Error", str(e))



    def start_trigger(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")], title="Select Model File")
        self.shots_path = filedialog.askdirectory(title="Select Shots Folder")
        if self.model_path and self.shots_path:
            self.tabview.set("Blank")
            self.btNastaveni.configure(state="disabled")
            self.btFunkce2.configure(state="disabled")
            self.btFunkce3.configure(state="disabled")
            self.btFunkce4.configure(state="disabled")
            self.btFunkce5.configure(state="disabled")
            self.btFunkce6.configure(state="disabled")
            threading.Thread(target=self.run_trigger, args=(self.model_path, self.shots_path), daemon=True).start()

    def run_trigger(self, model_path, shots_path):
        self.detector = PhoneDetector(model_path=model_path)
        self.detector.start()

    def stop_trigger(self):
        self.detector.stop()
        self.tabview.set("Webcam")
        self.btNastaveni.configure(state="normal")
        self.btFunkce2.configure(state="normal")
        self.btFunkce3.configure(state="normal")
        self.btFunkce4.configure(state="normal")
        self.btFunkce5.configure(state="normal")
        self.btFunkce6.configure(state="normal")

    def on_closing(self):
        self.save_variables()
        self.video_thread.join()
        self.camera.stop()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
