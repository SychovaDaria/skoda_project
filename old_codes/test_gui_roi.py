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
from edges import EdgeDetector
from typing import List

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

        #self.roi_settings = [] # TODO: for now, uniform and default
        self.drawn_lines = []


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
        for line in self.drawn_lines:
            self.canvas.delete(line)
        img_height,img_width,_ = np.shape(img)
        for roi in self.roi_list:
            cur_roi_coords = [int(i) for i in self.canvas.coords(roi)]
            x1,y1,x2,y2 = self.canvas_to_img_coords(cur_roi_coords, img_width, img_height)
            image = img[y1:y2,x1:x2,:]
            lines = self.start_edge_detection(image)
            for line in lines:
                print("Line",line)
                x0,y0,x1,y2 = self.img_to_canvas_coords(line, img_width, img_height) + np.array(np.concatenate((cur_roi_coords[0:2],cur_roi_coords[0:2])))
                can_line = self.canvas.create_line(x0,y0,x1,y2)
                self.drawn_lines.append(can_line)
        img = cv2.resize(img, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_img_ref = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(img)) 
        # if i dont have the reference, garbage colector will delete the img and canvas will show jacksh*t :)
        self.canvas.itemconfigure(self.background_img, image=self.current_img_ref)

        self.after(10, self.update_video_stream)

    def start_edge_detection(self, image):
        edge = EdgeDetector(min_val=10, max_val=50, min_value_of_votes=90, 
                                 min_length_of_straight_line=60,min_length=60,max_gap_between_lines=4,
                                 angle=90,angle_tolerance=180)
        return edge.get_lines(image)
        
   
 
    def canvas_to_img_coords(self, coords:List[int], img_width:int, img_height:int) -> List[int]:
        x1,y1,x2,y2 = coords
        can_width = self.canvas.winfo_width()
        can_height = self.canvas.winfo_height()
        x1 = int(np.floor(x1/can_width*img_width))
        x2 = int(np.floor(x2/can_width*img_width))
        y1 = int(np.floor(y1/can_height*img_height))
        y2 = int(np.floor(y2/can_height*img_height))
        return [x1,y1,x2,y2]
    
    def img_to_canvas_coords(self, coords:List[int], img_width:int, img_height:int) -> List[int]:
        x1,y1,x2,y2 = coords
        can_width = self.canvas.winfo_width()
        can_height = self.canvas.winfo_height()
        x1 = int(np.floor(x1/img_width*can_width))
        x2 = int(np.floor(x2/img_width*can_width))
        y1 = int(np.floor(y1/img_height*can_height))
        y2 = int(np.floor(y2/img_height*can_height))
        return [x1,y1,x2,y2]



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
