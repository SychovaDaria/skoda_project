import numpy as np 
from PIL import Image 
from glob import glob 
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from random import shuffle
import pickle

"""
Function to downsample an image for training the global classification model
Arguments
path: Image Path on device 
nw: Width of downsampled image
nh: Height of downsample image
Return: Downsampled numpy array
"""
def getimage_global(path, nw, nh):
    img = Image.open(path)
    img = img.resize([nw, nh], Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
    img = np.array(img, dtype=np.float32)
    img = np.divide(img, 255.0)
    img = np.reshape(img, (nw, nh, 1))
    return img

"""
Function to allow user to select the manual points for image extraction. 
Arguments
imgpath: Image Path on device
imgsetx: List in which the point would be added for future use
imgno: Image number
gw: Width of original Image
gh: Height of original Image
nw: Width of downsampled Image
nh: Height of downsampled Image
"""
def getpatches_sub(imgpath, imgsetx, imgno, gw, gh, nw, nh):
    img = Image.open(imgpath)
    img = np.array(img, dtype=np.float32)
    img = np.divide(img, 255.0)
    img = np.reshape(img, (gw, gh, 1))
    pltimg = plt.imread(imgpath)
    ax = plt.gca()
    fig = plt.gcf()
    ax.imshow(pltimg, cmap="gray")
    text_count = ax.text(0, 0, "count= 0")
    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, ax, text_count, imgno, imgsetx, gw, gh, nw, nh))
    plt.show()
    return fig

"""
An onclick image listener. Saves the extraction point in list. 
Arguments
event: mouse click event
ax: image plot axis
text_count: Text Header of the image
imgno: Image number
imgsetx: List in which the point would be added for future use
gw: Width of original Image
gh: Height of original Image
nw: Width of downsampled Image
nh: Height of downsampled Image
"""
def onclick(event, ax, text_count, imgno, imgsetx, gw, gh, nw, nh):
    global cimgs
    xst = int(event.xdata - (nh / 2))
    yst = int(event.ydata - (nw / 2))
    if xst < 0:
        xst = 0
    if yst < 0:
        yst = 0
    if xst <= (gw - nw) and yst <= (gh - nh):
        rect = Rectangle((xst, yst), nw, nh, alpha=0.3)
        ax.add_patch(rect)
        ax.figure.canvas.draw()
        imgpoint = [xst, yst]
        imgsetx.append(imgpoint)
        cimgs += 1
        rect.remove()
        text_count.set_text(f"count= {cimgs} len= {imgno}")

"""
Function to prepare the dataset for training global and sub classification models.
Arguments
data_path: Path where class folders are located. 
classes: No. of product classes. 
gw: Width of original Image
gh: Height of original Image
nw: Width of downsampled Image
nh: Height of downsampled Image
tr_range_nd: no. of images to be included in training set for non-defective images per class
tr_range_d: no. of images to be included in training set for defective images per class
nsubpatches: no.of manual extractions on a defective image
"""
def collect_imgs(data_path, classes, gw, gh, nw, nh, tr_range_nd, tr_range_d, nsubpatches):
    global cimgs
    db = {}
    tr_x = []
    tr_y = []
    val_x = []
    val_y = []
    print('collecting images')
    for i in range(classes):
        db = {}
        cl_nd = glob(data_path + (f"Class{i+1}/*"))  # non-defective images
        cl_d = glob(data_path + (f"Class{i+1}_def/*"))  # defective images
        perm_nd = np.random.permutation(len(cl_nd))
        perm_d = np.random.permutation(len(cl_d))
        k = 0
        subtrx = []
        subvalx = []
        for k in range(len(perm_nd)):
            if k < tr_range_nd:
                tr_x.append(getimage_global(cl_nd[perm_nd[k]], nw, nh))
                tr_y.append(i)
            else:
                val_x.append(getimage_global(cl_nd[perm_nd[k]], nw, nh))
                val_y.append(i)
        print(f'got data of non-defective set for class {i+1} length={k}')

        k = 0
        for k in range(len(perm_d)):
            if k < tr_range_d:
                tr_x.append(getimage_global(cl_d[perm_d[k]], nw, nh))
                tr_y.append(i)
                imgsetx = []
                cimgs = 0
                fig = getpatches_sub(cl_d[perm_d[k]], imgsetx, k, gw, gh, nw, nh)
                while plt.fignum_exists(fig.number):
                    plt.pause(20)
                shuffle(imgsetx)
                imgsetx = imgsetx[:nsubpatches]
                subtrx.append(imgsetx)
            else:
                val_x.append(getimage_global(cl_d[perm_d[k]], nw, nh))
                val_y.append(i)
                imgsetx = []
                cimgs = 0
                fig = getpatches_sub(cl_d[perm_d[k]], imgsetx, k, gw, gh, nw, nh)
                while plt.fignum_exists(fig.number):
                    plt.pause(20)
                shuffle(imgsetx)
                imgsetx = imgsetx[:nsubpatches]
                subvalx.append(imgsetx)
        print(f"got data of defective set for class={i+1} length={k}")

        subtrx = np.array(subtrx)
        subvalx = np.array(subvalx)
        db['ndefperm'] = perm_nd
        db['defperm'] = perm_d
        db[f'deftrp{i+1}'] = subtrx
        db[f'defvalp{i+1}'] = subvalx
        print(f"for sub class= {i+1} subtrx= {subtrx.shape} subvalx= {subvalx.shape}")
        file_path = data_path + (f"qcdlsub{i+1}list")
        with open(file_path, "wb") as dbfile:
            pickle.dump(db, dbfile)
        db = {}

    tr_x = np.array(tr_x)
    tr_y = np.array(tr_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    print("global", tr_x.shape, tr_y.shape, val_x.shape, val_y.shape)
    permutation = np.random.permutation(tr_x.shape[0])
    tr_x = tr_x[permutation]
    tr_y = tr_y[permutation]
    db['tr_x'] = tr_x
    db['tr_y'] = tr_y
    db['val_x'] = val_x
    db['val_y'] = val_y
    file_path = data_path + "qcdlglobal"
    with open(file_path, "wb") as dbfile:
        pickle.dump(db, dbfile)


cimgs = 0
classes = 12           # no. of classes of products
gw = 512              # width original image
gh = 512              # height of original image
nh = 128              # width after preprocessing
nw = 128              # height after preprocessing
tr_range_nd = 700     # no. of images to be included in training set for non-defective images per class
tr_range_d = 106      # no. of images to be included in training set for defective images per class
nsubpatches = 22      # no.of manual extractions on a defective image
data_path = "./clases/"  # path where class folders are located
collect_imgs(data_path, classes, gw, gh, nw, nh, tr_range_nd, tr_range_d, nsubpatches)


    
