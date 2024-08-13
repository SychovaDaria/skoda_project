"""
Leader process for the whole application.

This is the master process that will start all the other processes and manage them, namely GUI, the AI module,
and the trigger (image aquisition) module. Each slave process comunicates with other processes via queues, ie. the
settings and result queue.

The GUI process has 2 settings queues, one for the AI module and one for the trigger module. The AI module settings 
contain information such as the folders of training data, etc. The trigger module settings contain information about
the number of images to capture, etc. The GUI process also has a data queue, where the AI module puts the results of
the image processing.

On the end of the program, the GUI process will send a END message using both settings queues to the AI and trigger, which will 
break their loops and join the main process.

Author: Josef Kahoun
Date: 7. 8. 2024
"""

import multiprocessing
from trigger import Trigger
from GUI_ttk import App
from ttoo3gui import PhoneDetector
from process_settings import AiSettings

new_settings_ready = multiprocessing.Condition()

END_PROCESS_MESSAGE = "END"


def start_gui_process(img_queue, result_queue, settings_queue_ai, settings_queue_trigger):
    """
    Starts the GUI process.

    Args:
        img_queue (multiprocessing.Queue): Queue for the images from the AI process.
        result_queue (multiprocessing.Queue): Queue for the results from the AI process.
        settings_queue_ai (multiprocessing.Queue): Queue for the settings for the AI process.
        settings_queue_trigger (multiprocessing.Queue): Queue for the settings for the trigger process.
    """
    print("START GUI")
    app = App(img_queue,result_queue, settings_queue_ai, settings_queue_trigger)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
    print("END GUI")

def start_ai_trigger(img_queue, result_queue, settings_queue_ai):
    """
    Starts the AI and trigger processes.

    Args:
        img_queue (multiprocessing.Queue): Queue for the images from the AI process.
        result_queue (multiprocessing.Queue): Queue for the results from the AI process.
        settings_queue_ai (multiprocessing.Queue): Queue for the settings for the AI process.
    """
    print("START AI")
    img = None
    settings = AiSettings()
    settings.update_settings({'conf_thr': 0.7, 'model_path': 'best_model_state_dict_f12.pth'}) #FIXME: bad model
    detector = PhoneDetector(model_path=settings.model_path, img_height=150, img_width=150, capture_interval=20)
    while True:
        # load the last img
        while not img_queue.empty():
            img = img_queue.get() 
        # load last uploaded settings
        while not settings_queue_ai.empty():
            settings = settings_queue_ai.get()
        # check if the end message was sent
        if settings.is_end_message():
            break
        #TODO: load the settings for the AI module
        detector.unpack_settings(settings) 
        # check if the trigger should be started
        if True:#settings.is_start_trigger(): #FIXME: ADD TO GUI
            result = detector.detect_phone(img)
            if result:
                result_queue.put(result)
    print("END AI")

if __name__ == "__main__":
    # define the queues
    img_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    settings_queue_ai = multiprocessing.SimpleQueue()
    settings_queue_trigger = multiprocessing.SimpleQueue()
    gui_process = multiprocessing.Process(target=start_gui_process, args=(img_queue,result_queue, settings_queue_ai, settings_queue_trigger))
    ai_process = multiprocessing.Process(target=start_ai_trigger, args=(img_queue, result_queue, settings_queue_ai))
    # start the processes
    gui_process.start()
    #ai_process.start()
    # join the processes
    gui_process.join()
    #ai_process.join()
    print("END MAIN")   