"""
File for trying out the block (Scratch like) interface in ttk.
"""
import tkinter as tk

BLOCK_X_GRID = 150
BLOCK_Y_GRID = 50

class Block:
    def __init__(self, canvas, x, y, width, height, text):
        self.canvas = canvas
        self.block = canvas.create_rectangle(x, y, x + width, y + height, fill="lightblue", outline="black")
        self.label = canvas.create_text(x + width / 2, y + height / 2, text=text)
        self.drag_data = {"x": 0, "y": 0}
        self.canvas.tag_bind(self.block, "<ButtonPress-1>", self.on_click)
        self.canvas.tag_bind(self.label, "<ButtonPress-1>", self.on_click)
        self.canvas.tag_bind(self.block, "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(self.label, "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(self.block, "<ButtonRelease-1>", self.snap_to_grid)
        self.canvas.tag_bind(self.label, "<ButtonRelease-1>", self.snap_to_grid)

    def on_click(self, event):
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_drag(self, event):
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        self.canvas.move(self.block, dx, dy)
        self.canvas.move(self.label, dx, dy)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def snap_to_grid(self, event):
        block_coords = self.canvas.coords(self.block)
        x1, y1, _, _ = block_coords
        # Define grid snapping 
        snap_x = round(x1 / BLOCK_X_GRID) * BLOCK_X_GRID
        snap_y = round(y1 / BLOCK_Y_GRID) * BLOCK_Y_GRID
        # Move the block to the nearest grid position
        self.canvas.move(self.block, snap_x - x1, snap_y - y1)
        self.canvas.move(self.label, snap_x - x1, snap_y - y1)
        # TODO: check position and add connections

        


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scratch-like Interface")

        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=0, column=0, sticky="nsew")
        self.add_block_button = tk.Button(self.button_frame, text="Add block")
        self.add_block_button.pack()
        self.add_block_button.bind("<Button-1>", self.add_block)
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew")
        self.canvas = tk.Canvas(self.canvas_frame, bg="lightgray", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # ADD moving

        # Create some blocks
        block1 = Block(self.canvas, 50, 50, 150, 50, "Move 10 steps")
        block2 = Block(self.canvas, 50, 120, 150, 50, "Turn 90 degrees")
        self.blocks = [block1, block2]
        self.bind("<KeyPress>", self.on_key_press)

    def on_key_press(self, event):
        key = event.keysym
        if key == "Up":
            dx = 0
            dy = -BLOCK_Y_GRID
        elif key == "Down":
            dx = 0
            dy = BLOCK_Y_GRID
        elif key == "Left":
            dx = -BLOCK_X_GRID
            dy = 0
        else:
            dx = BLOCK_X_GRID
            dy = 0
        for block in self.blocks:
            x1,y1,_,_ = self.canvas.coords(block.block)
            self.canvas.move(block.block, dx, dy)
            self.canvas.move(block.label,dx, dy)

    def add_block(self, event) -> None:
        """
        Adds a block to the canvas.

        Args:
            None

        Returns:
            None
        """
        block = Block(self.canvas, 50, 50, 150, 50, "Block")
        self.blocks.append(block)

    def on_closing(self) -> None:
        """
        Closes the GUI, stops the camera and saves the settings variables into a file.

        Args:
            None

        Returns:
            None
        """
        self.root.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
