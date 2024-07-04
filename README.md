# skoda_project
raspberri camery a tak dal

# Guys 
archive contains photos of mobile phones for model. Saved model is in .h5 file. Model is not correct but it should work, tensor.py uses the model and suppose to turn on a stream and if it sees mobile phone it should take a photo. 

You also can write here notes so all of us know what do we do :)

# Coppying codes to raspberry pi
 
Using *scp* function in terminal. You need to be connected to the same network as the raspberry. <br/>
<pre>
    scp *file path in your computer* *raspberry_username*@*raspberry_IP_address*:~/*path in raspberry*
</pre>
In our case(example with raspicam.py module located in the directory, where i am calling this)
<pre>
    scp raspicam.py rpi@192.168.1.24:~/project/raspicam.py
</pre>
You will be prompted to enter password, the current one is *slunce*


# raspicam module
I created a docstr repo for documentation, most of the information about the module is in the raspicam.html file (exported docstrings). In short, the module can:
- drive the raspberry camera module, or an USB camera
- use different settings (all are saved as atributes)
- collect pictures in form of numpy array
- save pictures in the desired folder

# trigger module
A module for taking the pictures with the .trigg() function. You have to provide a camera (Raspicam object) and a folder name where to save the pictures, then you can set:
- number of pictures
- the delay between the trigger and the aquisition of the first picture
- the delay between the capture of different pictures, either as a float (every delay will then be the same), or as a list of floats (with length num_of_pictures - 1) that will specify each delay

Everything is specified in the docstring. 
IT WILL NEED IT'S OWN THREAD, because the module uses time.sleep() for the delay, which freezes everything else (if we don't want to use threads, need to replace the sleep() function with something that doesn't freeze everything, but this is the easiest way). <br />
When it is set, we can call the .trig() when an event occurs (GPIO, color blobs, edge detection, AI, ...)

# Multithreading
## What will need its own thread?
- GUI - stream
- GUI - settings
- computation --> AI, edges, blobs, ...
- the trigger module

## Critical sections ?
- ¯\\_(ツ)_/¯

## Example
Example shown in src/thread_test.py (je to celkem ez, jediny co, tak hlidat, jestli nekde neni kriticka sekce (napr. kdyz jsou dva thready, jeden do sdilene promenne zapisuje a druhy z ni cte, pak je to undefined behavior), pak jsou treba mutexy/cond var, ale v tom pythonu jsou hezky na radek vytvoreny)

# GPIO inputs
via push_button.py

# Pepa tasks:
Threads
Raspicam auto brightness just based on ROI
Remote controled raspberry
