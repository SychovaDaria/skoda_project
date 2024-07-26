import PIL.Image
import PIL.ImageTk
import customtkinter as ctk
import numpy as np
import tkinter
import threading
from raspicam import Raspicam
import cv2
import PIL
import time


CORNER_THRESHOLD = 10
NO_ROI_SELECTED = -1



class RoiSelector(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        self.geometry("%dx%d" % (width, height))
        self.minsize(400, 300)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.btnFrame = ctk.CTkFrame(self)
        self.btnFrame.grid(row=0, column=0, sticky="ew")
        self.btn = ctk.CTkButton(self.btnFrame, text="Create ROI", command=self.create_roi)
        self.btn.grid(row=0, column=0, sticky="ew")
        self.canvasFrame = ctk.CTkFrame(self)
        self.canvasFrame.grid(row=0, column=1, sticky="nsew", rowspan=2)
        self.statsFrame = ctk.CTkFrame(self)
        self.statsFrame.grid(row=1, column=0, sticky="ew")
        self.canvas = tkinter.Canvas(self.canvasFrame)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)


        self.statLbl = ctk.CTkLabel(self.statsFrame, text="Stats")
        self.statLbl.grid(row=0, column=0, sticky="ew")

        self.roi = []
        self.move_roi = False
        self.resize_roi_left = False
        self.resize_roi_right = False
        self.offset_x = 0   
        self.offset_y = 0
        self.roi_id = NO_ROI_SELECTED
        self.roi_list = []

        img = PIL.Image.open("../test_img/door.jpeg")
        img = img.resize((self.winfo_screenwidth(), self.winfo_screenheight()))
        photo_img = PIL.ImageTk.PhotoImage(img)
        self.current_img_ref = photo_img
        self.background_img = self.canvas.create_image(0, 0, image=self.current_img_ref, anchor = tkinter.NW)
        time.sleep(10)
        self.camera = Raspicam(use_usb=True)
        self.video_thread = threading.Thread(target=self.update_video_stream)
        self.video_thread.start()

    def create_roi(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        self.roi = [4*width//9, 4*height//9, 5*width//9, 5*height//9]
        self.roi_list.append(self.canvas.create_rectangle(self.roi, outline="red"))


    def update_stat_label(self):
        self.statLbl.configure(text="Stats"+"   "+str(self.roi_id))

    def update_video_stream(self):
        img = self.camera.capture_img()
        img = cv2.resize(img, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_img_ref = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(img))
        self.canvas.itemconfigure(self.background_img, image=self.current_img_ref)
        self.after(10, self.update_video_stream)

    def on_click(self, event):
        # check if i am clicking near the top left and bottom right corners of the roi
        for roi in self.roi_list:
            current_coords = self.canvas.coords(roi)
            if (np.linalg.norm(np.array([int(event.x), int(event.y)])-np.array(self.canvas.coords(roi)[:2])) < CORNER_THRESHOLD):
                self.resize_roi_left = True
                if roi != self.roi_id:
                    self.canvas.itemconfig(self.roi_id, outline="red")
                    self.roi_id = roi
                break 
            elif np.linalg.norm(np.array([event.x, event.y])-np.array(self.canvas.coords(roi)[2:])) < CORNER_THRESHOLD:
                self.resize_roi_right = True
                if roi != self.roi_id:
                    self.canvas.itemconfig(self.roi_id, outline="red")
                    self.roi_id = roi
                break
            elif event.x > current_coords[0] and event.x < current_coords[2] and event.y > current_coords[1] and event.y < current_coords[3]:
                self.move_roi = True
                self.offset_x = event.x - current_coords[0]
                self.offset_y = event.y - current_coords[1]
                if roi != self.roi_id:
                    self.canvas.itemconfig(self.roi_id, outline="red")
                    self.roi_id = roi
                break
        if not (self.move_roi or self.resize_roi_left or self.resize_roi_right):
            self.canvas.itemconfig(self.roi_id, outline="red")
            self.roi_id = NO_ROI_SELECTED
        else:
            self.canvas.itemconfig(self.roi_id, outline="blue")
        self.update_stat_label()



    def on_drag(self, event):
        if self.resize_roi_left:
            self.canvas.coords(self.roi_id, event.x, event.y, self.canvas.coords(self.roi_id)[2], self.canvas.coords(self.roi_id)[3])
        elif self.resize_roi_right:
            self.canvas.coords(self.roi_id, self.canvas.coords(self.roi_id)[0], self.canvas.coords(self.roi_id)[1], event.x, event.y)
        elif self.move_roi:
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            current_coords = self.canvas.coords(self.roi_id)
            dx = event.x - current_coords[0] - self.offset_x
            dy = event.y - current_coords[1] - self.offset_y
            if current_coords[0] + dx < 0 or current_coords[2] + dx > width:
                dx = 0
            if current_coords[1] + dy < 0 or current_coords[3] + dy > height:
                dy = 0
            self.canvas.move(self.roi_id, dx, dy)

    def on_release(self, event):
        if self.resize_roi_left or self.resize_roi_right:
            self.resize_roi_left = False
            self.resize_roi_right = False
        if self.move_roi:
            self.move_roi = False
        

if __name__ == "__main__":
    app = RoiSelector()
    app.mainloop()
