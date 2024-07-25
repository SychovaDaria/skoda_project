from GUI import App
import multiprocessing
import time
from raspicam import Raspicam
import numpy as np

end_app = False


def start_gui(data_queue, settings_queue):
    print("START GUI")
    app = App(data_queue,settings_queue)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
    print("END GUI")

def get_pictures(data_queue, settings_queue):
    print("START IMAGES")
    cam = Raspicam()
    last_image = []
    while True:
        if not settings_queue.empty():
            finish = settings_queue.get()
            if finish == True:
                #print("GET_PICTURES_BREAK")
                break
        # put the circles in the data queue
        img = cam.capture_img()
        comparison = np.array_equal(img, last_image)
        print(comparison)
        if not comparison:
            data_queue.put(img)
            last_image = img
            #print("put img in queue")
        time.sleep(0.015)
    print("END IMAGES")


def main():
    # start the gui process
    data_queue = multiprocessing.SimpleQueue()
    settings_queue = multiprocessing.SimpleQueue()
    gui_process = multiprocessing.Process(target=start_gui, args=(data_queue, settings_queue))
    comp_process = multiprocessing.Process(target=get_pictures, args=(data_queue, settings_queue))
    # start the computation process
    gui_process.start()
    comp_process.start()
    gui_process.join()
    comp_process.join()

    
    

if __name__ == "__main__":
    main()