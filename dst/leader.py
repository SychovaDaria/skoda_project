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
Date: 7. 08. 2024
"""

from GUI_ttk import App
import multiprocessing
from trigger import Trigger

def start_gui_process(data_queue, settings_queue_ai, settings_queue_trigger):
    """
    Starts the GUI process.

    Args:
        data_queue (multiprocessing.Queue): The queue where the GUI process will get the results from the AI module.
        settings_queue_ai (multiprocessing.Queue): The queue where the GUI process will send the settings to the AI module.
        settings_queue_trigger (multiprocessing.Queue): The queue where the GUI process will send the settings to the trigger module.
    """
    print("START GUI")
    app = App(data_queue, settings_queue_ai, settings_queue_trigger)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
    print("END GUI")

def start_ai_process(data_queue, settings_queue_ai):
    pass

def start_trigger_process(data_queue, settings_queue_trigger):
    trigger = Trigger()

if __name__ == "__main__":
    # start the gui process
    data_queue = multiprocessing.SimpleQueue()
    settings_queue_ai = multiprocessing.SimpleQueue()
    settings_queue_trigger = multiprocessing.SimpleQueue()
    gui_process = multiprocessing.Process(target=App, args=(data_queue, settings_queue_ai, settings_queue_trigger))
    # start the computation process
    gui_process.start()
    gui_process.join()