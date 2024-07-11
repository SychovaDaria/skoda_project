import cv2
import numpy as np
from raspicam import Raspicam
from ultralytics import YOLO
from typing import List, Tuple

model = YOLO("yolov8n.pt")
CONFIDENCE_THRESHOLD = 0.6

def get_people_boxes(img : np.array) -> List[Tuple[int,int,int,int]]:
    """
    Get the boxes of the people in the image.

    Args:
        img (np.array): The image to detect people in.

    Returns:
        list: List of tuples containing the x1, y1, x2, y2 coordinates of the people.
    """
    result = model(img, conf = CONFIDENCE_THRESHOLD)[0] # the yolo result
    summ = result.summary() #  list of dictionaries with name, confidence and box
    # go through each obejct and check if they are a person, then add their box to the list
    people_boxes = []
    for obj in summ:
        if obj['name'] == 'person':
            x1 = obj['box']['x1']
            y1 = obj['box']['y1']
            x2 = obj['box']['x2']
            y2 = obj['box']['y2']
            people_boxes.append((x1,y1,x2,y2))
    return people_boxes

def get_inteference_x(people_boxes : List[Tuple[int,int,int,int]], width : int) -> List[bool]:
    """
    Returns the inteference list of the x axis.

    Returns a binary list of lenght 2^6 where 1 means that there is a person in the corresponding part of the image.

    Args:
        people_boxes (list): List of people boxes.
        width (int): The width of the image.
    
    Returns:
        list: List of booleans representing the inteference in the x axis.
    """
    # create the list
    inteference = [False]*2**6
    # go through each person box and set the corresponding part of the inteference list to True
    for box in people_boxes:
        x1, _, x2, _ = box
        x1 = int(x1/width*(2**6))
        x2 = int(x2/width*(2**6))
        for i in range(x1,x2):
            inteference[i] = True
    return inteference


def get_inteference_xy(people_boxes : List[Tuple[int,int,int,int]], width : int, height : int) -> List[List[bool]]:
    """
    Returns the inteference matrix of the x and y axis.

    Returns a binary matrix of size 2^6x2^4 where 1 means that there is a person in the corresponding part of the image.
    
    Args:
        people_boxes (list): List of people boxes.
        width (int): The width of the image.
        height (int): The height of the image.
    
    Returns:
        list: List of lists of booleans representing the inteference in the x and y axis.
    """ 
    # create the matrix
    inteference = [[False]*2**6 for _ in range(2**4)]
    # go through each person box and set the corresponding part of the inteference matrix to True
    for box in people_boxes:
        x1, y1, x2, y2 = box
        x1 = int(x1/width*(2**6))
        x2 = int(x2/width*(2**6))
        y1 = int(y1/height*(2**4))
        y2 = int(y2/height*(2**4))
        for c in range(x1,x2):
            for r in range(y1,y2):
                inteference[r][c] = True
    return inteference

def plot_inteference(inteference : List[List[bool]] | List[bool], img: np.array) -> np.array:
    """
    Indicates the inteference on the current img

    Args:
        inteference (list): The inteference list or matrix.
        img (np.array): The image to plot the inteference on.
    
    Returns:
        np.array: The plotted image.
    """
    height, width, _ = img.shape
    # check if its the x or xy inteference
    if isinstance(inteference[0], bool): # the x inteference
        for i in range(len(inteference)):
            if inteference[i]:
                cv2.rectangle(img, (i*width//64,0), ((i+1)*width//64,height), (0,0,255), 2)
    else: # xy inteference
        for r in range(len(inteference)):
            for c in range(len(inteference[0])):
                if inteference[r][c]:
                    cv2.rectangle(img, (c*width//64,r*height//16), ((c+1)*width//64,(r+1)*height//16), (0,0,255), 2)
    return img

def main():
    cam = Raspicam()
    img = cam.capture_img()
    # get the shape of the image
    height, width, _ = img.shape
    while True:
        # get the img
        img = cam.capture_img()
        # the sensor on our camera is flipped
        img = cv2.flip(img, -1)
        # get the people boxes
        people_boxes = get_people_boxes(img)
        
        #inteference = get_inteference_x(people_boxes, width)
        inteference = get_inteference_xy(people_boxes, width, height)
        img = plot_inteference(inteference, img)
        
        cv2.imshow("Stream", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.stop()

if __name__ == "__main__":
    main()


