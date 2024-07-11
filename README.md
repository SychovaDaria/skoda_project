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
The GUI library has some sort of multithreading already implemented, so it is not recommended to try multithreading. However, multiprocessing can be implemented, the only restriction being the GUI has to be implemented in just one process.<br /> 
The app should be faster if we implement the computation modules (AI, edges, ...) into seperate processes. The communication between them can be utilized using a pipe/queue. Our raspberry pi has 4 cores, so max 4 processes, one of which will be the GUI.
Example shown in the tryouts directory. 

# GPIO inputs
via push_button.py

# YOLO navigation
tried with ultralytics in yolo_nav.py, there are functions there that take the bounding boxes of objects detected as 'person', then they are converted either to array containing 2^6 booleans (can be converted to one int using binary), or a matrix 2^6x2^4, which hold the information about position of people in the scene

# QR and barcode reading 
class implemented in codes_detector.py

# Stitching images
cv2.Stitcher_create(), but its not perfect. Good for creating panorama, but fe if we would have a moving part on a conveyer belt, then it would try to stitch it according to the background if it was visible an non-uniform (fe on all white background this is not a problem, but when there are a lot of features, it is). Could be fixed by either cropping all the images, or make a roi where the algorhytm would look for the features, but that would mean to write our own algo (img to grayscale, extract features from roi, knn, RANSAC, ..., but cv2 has most of this implemented so its not that hard)

# Connecting to rapsberry pi remotely
## Raspberry pi connect
Very easy way to do this, but since it goes through the raspberry servers we can't use it, but if you want to use it just for development, check the (https://www.raspberrypi.com/software/connect/ "Rasberry Pi Connect") webpage.
## VNC server
Start a VNC server on the raspberry and then connect through a VNC reader app, for example (https://www.realvnc.com/en/connect/download/viewer/?lai_sr=0-4&lai_sl=l, "RealVNC Viewer").
## NoVNC
This should be a way so no app is needed, you are supposed to just open a link in a browser, but I could not get it to work, here is the link to their github: (https://github.com/novnc/noVNC#browser-requirements)

# Pepa tasks:
- Threads
- Raspicam auto brightness just based on ROI
- Remote controled raspberry
- can you flash the sd card with the app already preinstalled ??