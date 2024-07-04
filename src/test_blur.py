from skda_blur import Filter
import cv2
import numpy as np

def main():
    img = cv2.imread("../test_img/blob_test.jpg")
    filter = Filter()
    img = filter.average(img)
    cv2.imshow("Average", img)
    img = cv2.imread("../test_img/blob_test.jpg")
    img = filter.gaussian(img)
    cv2.imshow("Gaussian", img)
    img = cv2.imread("../test_img/blob_test.jpg")
    img = filter.median(img)
    cv2.imshow("Median", img)
    img = cv2.imread("../test_img/blob_test.jpg")
    img = filter.bilateral(img)
    cv2.imshow("Bilateral", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()