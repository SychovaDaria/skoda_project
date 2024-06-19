# skoda_project
raspberri camery a tak dal

# Guys 
archive contains photos of mobile phones for model. Saved model is in .h5 file. Model is not correct but it should work, tensor.py uses the model and suppose to turn on a stream and if it sees mobile phone it should take a photo. 

You also can write here notes so all of us know what do we do :)

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
IT WILL NEED IT'S OWN THREAD, because the module uses time.sleep() for the delay, which freezes everything else (if we don't want to use threads, need to replace the sleep() function with something that doesn't freeze everything, but this is the easiest way)
When it is set, we can call the .trig() when an event occurs (GPIO, color blobs, edge detection, AI, ...)