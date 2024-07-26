import customtkinter as ctk
import numpy as np
import tkinter

CORNER_THRESHOLD = 10

class RoiSelector(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        self.geometry("%dx%d" % (width, height))
        self.minsize(400, 300)

        self.canvas = tkinter.Canvas(self, bg="white")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.roi = []
        self.move_roi = False
        self.resize_roi_left = False
        self.resize_roi_right = False
        self.roi_id = -1
        self.roi_list = []

    def create_roi(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        self.roi = [4*width//9, 4*height//9, 5*width//9, 5*height//9]
        self.roi_id = self.canvas.create_rectangle(self.roi, outline="red")
        self.roi_list.append(self.roi_id)
        print("Coords",self.canvas.coords(self.roi_id))

    def on_click_test(self, event):
        self.create_roi()
        print(self.roi_list, self.roi_id)
        print(self.canvas.coords(self.roi_id))



    def on_click(self, event):
        # check if i am clicking near the top left and bottom right corners of the roi
        for roi in self.roi_list:
            current_coords = self.canvas.coords(roi)
            print("Coords", current_coords)
            if (np.linalg.norm(np.array([int(event.x), int(event.y)])-np.array(self.canvas.coords(roi)[:2])) < CORNER_THRESHOLD):
                self.resize_roi_left = True
                self.roi_id = roi
                break 
            elif np.linalg.norm(np.array([event.x, event.y])-np.array(self.canvas.coords(roi)[2:])) < CORNER_THRESHOLD:
                self.resize_roi_right = True
                self.roi_id = roi
                break
        if not (self.resize_roi_left or self.resize_roi_right):
            self.create_roi()


    def on_drag(self, event):
        if self.resize_roi_left:
            self.canvas.coords(self.roi_id, event.x, event.y, self.canvas.coords(self.roi_id)[2], self.canvas.coords(self.roi_id)[3])
        elif self.resize_roi_right:
            self.canvas.coords(self.roi_id, self.canvas.coords(self.roi_id)[0], self.canvas.coords(self.roi_id)[1], event.x, event.y)
    
    def on_release(self, event):
        if self.resize_roi_left or self.resize_roi_right:
            self.resize_roi_left = False
            self.resize_roi_right = False
        if self.move_roi:
            self.move_roi = False

if __name__ == "__main__":
    app = RoiSelector()
    app.mainloop()
