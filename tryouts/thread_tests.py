from GUI import App
import multiprocessing
import time

end_app = False

def start_gui(data_queue,settings_queue):
    print("START GUI")
    app = App(data_queue,settings_queue)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

def start_circle_detection(data_queue,settings_queue):
    print("START COMP")
    while True:
        if not settings_queue.empty():
            finish = settings_queue.get()
            if finish == True:
                break
        # put the circles in the data queue
        data_queue.put("circles")
        time.sleep(0.1)
    print("END COMP")

def main():
    # start the gui process
    data_queue = multiprocessing.SimpleQueue()
    settings_queue = multiprocessing.SimpleQueue()
    gui_process = multiprocessing.Process(target=start_gui, args=(data_queue,settings_queue))
    comp_process = multiprocessing.Process(target=start_circle_detection, args=(data_queue,settings_queue))
    # start the computation process
    gui_process.start()
    comp_process.start()
    gui_process.join()
    comp_process.join()

    
    

if __name__ == "__main__":
    main()