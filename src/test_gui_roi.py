"""
Quick test to try ROI selection on the GUI.

This is a quick test to try out the ROI selection on the GUI. The user can create a ROI by clicking the button. The user can also move the ROI by clicking and dragging the ROI. The user can also resize the ROI by clicking and dragging the corners of the ROI. The user can also create multiple ROIs. The user can also save the settings for each ROI.

Author: Josef Kahoun
Date: 6.8.2024

Tasks:
    - Create a thread for each ROI trigger.
    - add the wait after trigger delay
"""


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


class RoiSettings():
    """
    Class for storing settings for each ROI.

    Attributes:
        roi_id (int): The canvas id of the ROI.
        num_of_pictures (int): The number of pictures to take.
        delay_between_pictures (int): The delay between taking pictures in ms.
        first_delay (int): The delay before taking the first picture in ms.
        after_delay (int): The delay after taking the last picture in ms.
        min_val (int): The minimum value for the edge detector.
        max_val (int): The maximum value for the edge detector.
        min_value_of_votes (int): The minimum value of votes for the edge detector.
        min_length_of_straight_line (int): The minimum length of straight line for the edge detector.
        max_gap_between_lines (int): The maximum gap between lines for the edge detector.
        angle (int): The angle for the edge detector.
        angle_tolerance (int): The angle tolerance for the edge detector.
    """
    def __init__(self) -> None:
        self.roi_id = NO_ROI_SELECTED
        self.num_of_pictures = 1
        self.delay_between_pictures = 0
        self.first_delay = 0
        self.after_delay = 0
        self.min_val = 10
        self.max_val = 50
        self.min_value_of_votes = 90
        self.min_length_of_straight_line = 60
        self.max_gap_between_lines = 4
        self.angle = 0
        self.angle_tolerance = 10


    def update_settings(self, num_of_pictures: int, delay_between_pictures:int, first_delay:int, after_delay:int, min_val: int, max_val : int,
                        min_value_of_votes: int,min_length_of_straight_line: int, max_gap_between_lines: int, angle: int,
                        angle_tolerance: int) -> None:
        """
        Function to update the settings for the ROI.

        Args:
            num_of_pictures (int): The number of pictures to take.
            delay_between_pictures (int): The delay between taking pictures in ms.
            first_delay (int): The delay before taking the first picture in ms.
            after_delay (int): The delay after taking the last picture in ms.
            min_val (int): The minimum value for the edge detector.
            max_val (int): The maximum value for the edge detector.
            min_value_of_votes (int): The minimum value of votes for the edge detector.
            min_length_of_straight_line (int): The minimum length of straight line for the edge detector.
            max_gap_between_lines (int): The maximum gap between lines for the edge detector.
            angle (int): The angle for the edge detector.
            angle_tolerance (int): The angle tolerance for the edge detector.

        Returns:
            None
        """
        self.num_of_pictures = num_of_pictures
        self.delay_between_pictures = delay_between_pictures
        self.first_delay = first_delay
        self.after_delay = after_delay
        self.min_val = min_val
        self.max_val = max_val
        self.min_value_of_votes = min_value_of_votes
        self.min_length_of_straight_line = min_length_of_straight_line
        self.max_gap_between_lines = max_gap_between_lines
        self.angle = angle
        self.angle_tolerance = angle_tolerance




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
        
        

        self.settingsSaveBtn = ctk.CTkButton(self.btnFrame, text="Save settings", command=self.save_settings)
        self.settingsSaveBtn.grid(row=23, column=0, sticky="ew")
        
        
        self.canvasFrame = ctk.CTkFrame(self)
        self.canvasFrame.grid(row=0, column=1, sticky="nsew", rowspan=2)
        self.statsFrame = ctk.CTkFrame(self)
        self.statsFrame.grid(row=1, column=0, sticky="ew")
        self.canvas = tkinter.Canvas(self.canvasFrame)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)


        self.statLbl = ctk.CTkLabel(self.statsFrame, text="Current ROI ID:")
        self.statLbl.grid(row=0, column=0, sticky="ew")

        self.roi = []
        self.move_roi = False
        self.resize_roi_left = False
        self.resize_roi_right = False
        self.offset_x = 0   
        self.offset_y = 0
        self.roi_id = NO_ROI_SELECTED
        self.roi_list = []
        self.roi_settings_list = []

        #self.roi_settings = [] # TODO: for now, uniform and default
        self.drawn_lines = []

        self.camera = Raspicam(use_usb=True)
        img = self.camera.capture_img()
        img = cv2.resize(img, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_img_ref = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(img)) 
        self.background_img = self.canvas.create_image(0, 0, image=self.current_img_ref, anchor = tkinter.NW)
        
        self.video_thread = threading.Thread(target=self.update_video_stream)
        self.video_thread.start()

        

    def open_roi_settings_window(self):
        self.roi_settings_window = ctk.CTkToplevel(self, title="ROI settings")
        self.roi_settings_window.geometry("400x400")
        self.roi_settings_window.minsize(400, 400)
        
        self.settingsNumOfPicturesLbl = ctk.CTkLabel(self.roi_settings_window, text="Number of pictures")
        self.settingsNumOfPicturesLbl.grid(row=1, column=0, sticky="ew")
        self.settingsNumOfPictures = ctk.CTkEntry(self.roi_settings_window, placeholder_text="Number of pictures")
        self.settingsNumOfPictures.grid(row=1, column=1, sticky="ew")
        self.settingsNumOfPictures.insert(0, "1")
        self.settingsDelayBetweenPicturesLbl = ctk.CTkLabel(self.btnFrame, text="Delay between pictures")
        self.settingsDelayBetweenPicturesLbl.grid(row=2, column=0, sticky="ew")
        self.settingsDelayBetweenPictures = ctk.CTkEntry(self.btnFrame, placeholder_text="Delay between pictures")
        self.settingsDelayBetweenPictures.grid(row=2, column=1, sticky="ew")
        self.settingsDelayBetweenPictures.insert(0, str(self.settings))
        self.settingsFirstDelayLbl = ctk.CTkLabel(self.btnFrame, text="First delay")
        self.settingsFirstDelayLbl.grid(row=5, column=0, sticky="ew")
        self.settingsFirstDelay = ctk.CTkEntry(self.btnFrame, placeholder_text="First delay")
        self.settingsFirstDelay.grid(row=6, column=0, sticky="ew")
        self.settingsFirstDelay.insert(0, "0")
        self.settingsAfterDelayLbl = ctk.CTkLabel(self.btnFrame, text="After delay")
        self.settingsAfterDelayLbl.grid(row=7, column=0, sticky="ew")
        self.settingsAfterDelay = ctk.CTkEntry(self.btnFrame, placeholder_text="After delay")
        self.settingsAfterDelay.grid(row=8, column=0, sticky="ew")
        self.settingsAfterDelay.insert(0, "0")
        self.settingsMinValLbl = ctk.CTkLabel(self.btnFrame, text="Min val")
        self.settingsMinValLbl.grid(row=9, column=0, sticky="ew")
        self.settingsMinVal = ctk.CTkEntry(self.btnFrame, placeholder_text="Min val")
        self.settingsMinVal.grid(row=10, column=0, sticky="ew")
        self.settingsMinVal.insert(0, "10")
        self.settingsMaxValLbl = ctk.CTkLabel(self.btnFrame, text="Max val")
        self.settingsMaxValLbl.grid(row=11, column=0, sticky="ew")
        self.settingsMaxVal = ctk.CTkEntry(self.btnFrame, placeholder_text="Max val")
        self.settingsMaxVal.grid(row=12, column=0, sticky="ew")
        self.settingsMaxVal.insert(0, "50")
        self.settingsMinValueOfVotesLbl = ctk.CTkLabel(self.btnFrame, text="Min value of votes")
        self.settingsMinValueOfVotesLbl.grid(row=13, column=0, sticky="ew")
        self.settingsMinValueOfVotes = ctk.CTkEntry(self.btnFrame, placeholder_text="Min value of votes")
        self.settingsMinValueOfVotes.grid(row=14, column=0, sticky="ew")
        self.settingsMinValueOfVotes.insert(0, "90")
        self.settingsMinLengthOfStraightLineLbl = ctk.CTkLabel(self.btnFrame, text="Min length of straight line")
        self.settingsMinLengthOfStraightLineLbl.grid(row=15, column=0, sticky="ew")
        self.settingsMinLengthOfStraightLine = ctk.CTkEntry(self.btnFrame, placeholder_text="Min length of straight line")
        self.settingsMinLengthOfStraightLine.grid(row=16, column=0, sticky="ew")
        self.settingsMinLengthOfStraightLine.insert(0, "60")
        self.settingsMaxGapBetweenLinesLbl = ctk.CTkLabel(self.btnFrame, text="Max gap between lines")
        self.settingsMaxGapBetweenLinesLbl.grid(row=17, column=0, sticky="ew")
        self.settingsMaxGapBetweenLines = ctk.CTkEntry(self.btnFrame, placeholder_text="Max gap between lines")
        self.settingsMaxGapBetweenLines.grid(row=18, column=0, sticky="ew")
        self.settingsMaxGapBetweenLines.insert(0, "4")
        self.settingsAngleLbl = ctk.CTkLabel(self.btnFrame, text="Angle")
        self.settingsAngleLbl.grid(row=19, column=0, sticky="ew")
        self.settingsAngle = ctk.CTkEntry(self.btnFrame, placeholder_text="Angle")
        self.settingsAngle.grid(row=20, column=0, sticky="ew")
        self.settingsAngle.insert(0, "0")
        self.settingsAngleToleranceLbl = ctk.CTkLabel(self.btnFrame, text="Angle tolerance")
        self.settingsAngleToleranceLbl.grid(row=21, column=0, sticky="ew")
        self.settingsAngleTolerance = ctk.CTkEntry(self.btnFrame, placeholder_text="Angle tolerance")
        self.settingsAngleTolerance.grid(row=22, column=0, sticky="ew")
        self.settingsAngleTolerance.insert(0, "10")
        



    def create_roi(self):
        """
        Function for creating the ROI, called by clicking the button.
        """
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        self.roi = [4*width//9, 4*height//9, 5*width//9, 5*height//9]
        self.roi_list.append(self.canvas.create_rectangle(self.roi, outline="red"))
        self.roi_settings_list.append(RoiSettings())


    def update_stat_label(self):
        if self.roi_id == NO_ROI_SELECTED:
            self.statLbl.configure(text="Current ROI ID:"+"   "+"None")
        else:
            self.statLbl.configure(text="Current ROI ID:"+"   "+str(self.roi_id))
        if self.roi_id != NO_ROI_SELECTED:
            self.load_settings()

    def update_video_stream(self):
        img = self.camera.capture_img()
        for line in self.drawn_lines:
            self.canvas.delete(line)
        img_height,img_width,_ = np.shape(img)
        for roi in self.roi_list:
            cur_roi_coords = [int(i) for i in self.canvas.coords(roi)]
            x1,y1,x2,y2 = self.canvas_to_img_coords(cur_roi_coords, img_width, img_height)
            image = img[y1:y2,x1:x2,:]
            lines = self.start_edge_detection(image, self.roi_settings_list[self.roi_list.index(roi)])
            for line in lines:
                x0,y0,x1,y2 = self.img_to_canvas_coords(line, img_width, img_height) + np.array(np.concatenate((cur_roi_coords[0:2],cur_roi_coords[0:2])))
                can_line = self.canvas.create_line(x0,y0,x1,y2)
                self.drawn_lines.append(can_line)
        img = cv2.resize(img, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_img_ref = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(img)) 
        # if i dont have the reference, garbage colector will delete the img and canvas will show jacksh*t :)
        self.canvas.itemconfigure(self.background_img, image=self.current_img_ref)

        self.after(400, self.update_video_stream)

    def start_edge_detection(self, image, settings):
        edge = EdgeDetector(min_val=settings.min_val, max_val=settings.max_val, min_value_of_votes=settings.min_value_of_votes, 
                                 min_length_of_straight_line=settings.min_length_of_straight_line ,min_length=settings.min_length_of_straight_line,
                                 max_gap_between_lines=settings.max_gap_between_lines, angle=settings.angle,
                                 angle_tolerance=settings.angle_tolerance)
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
        
    def load_settings(self) -> None:
        """
        Loads the settings into the entry widgets for current ROI.

        Returns:
            None
        """
        index = self.roi_list.index(self.roi_id)
        settings = self.roi_settings_list[index]
        self.settingsNumOfPictures.delete(0, tkinter.END)
        self.settingsNumOfPictures.insert(0, str(settings.num_of_pictures))
        self.settingsDelayBetweenPictures.delete(0, tkinter.END)
        self.settingsDelayBetweenPictures.insert(0, str(settings.delay_between_pictures))
        self.settingsFirstDelay.delete(0, tkinter.END)
        self.settingsFirstDelay.insert(0, str(settings.first_delay))
        self.settingsAfterDelay.delete(0, tkinter.END)
        self.settingsAfterDelay.insert(0, str(settings.after_delay))
        self.settingsMinVal.delete(0, tkinter.END)
        self.settingsMinVal.insert(0, str(settings.min_val))
        self.settingsMaxVal.delete(0, tkinter.END)
        self.settingsMaxVal.insert(0, str(settings.max_val))
        self.settingsMinValueOfVotes.delete(0, tkinter.END)
        self.settingsMinValueOfVotes.insert(0, str(settings.min_value_of_votes))
        self.settingsMinLengthOfStraightLine.delete(0, tkinter.END)
        self.settingsMinLengthOfStraightLine.insert(0, str(settings.min_length_of_straight_line))
        self.settingsMaxGapBetweenLines.delete(0, tkinter.END)
        self.settingsMaxGapBetweenLines.insert(0, str(settings.max_gap_between_lines))
        self.settingsAngle.delete(0, tkinter.END)
        self.settingsAngle.insert(0, str(settings.angle))
        self.settingsAngleTolerance.delete(0, tkinter.END)
        self.settingsAngleTolerance.insert(0, str(settings.angle_tolerance))

    def save_settings(self):
        """
        Saves the settings for current ROI.
        """
        index = self.roi_list.index(self.roi_id)
        settings = self.roi_settings_list[index]
        settings.update_settings(int(self.settingsNumOfPictures.get()), int(self.settingsDelayBetweenPictures.get()), int(self.settingsFirstDelay.get()), int(self.settingsAfterDelay.get()), 
                                 int(self.settingsMinVal.get()), int(self.settingsMaxVal.get()), int(self.settingsMinValueOfVotes.get()), int(self.settingsMinLengthOfStraightLine.get()),
                                 int(self.settingsMaxGapBetweenLines.get()), int(self.settingsAngle.get()), int(self.settingsAngleTolerance.get()))


if __name__ == "__main__":
    app = RoiSelector()
    app.mainloop()
