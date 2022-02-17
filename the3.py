import numpy as np
import cv2 as cv
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.io import imread
from skimage import segmentation, color
from skimage.future import graph

B1 = './THE3-Images/B1.png'
B2 = './THE3-Images/B2.png'
B3 = './THE3-Images/B3.png'
output_path = './THE3_outputs/'


def gammaCorrection(src, gamma):
    invGamma = gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv.LUT(src, table)


################### A1.PNG ###################
a1 = cv.imread('./THE3-Images/A1.png', 0)
a1 = gammaCorrection(a1, 1.1)
thresh = cv.adaptiveThreshold(a1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 235, 68)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (29, 29))
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

num, labels = cv.connectedComponents(closing)
print("The number of flying balloons in image A1 is", num - 1)
cv.imwrite("part1_A1.png", closing)

################### A2.PNG ###################
a2 = cv.imread('THE3-Images/A2.png', 0)
thresh = cv.adaptiveThreshold(a2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 255, 90)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

num, labels = cv.connectedComponents(closing)
print("The number of flying balloons in image A2 is", num - 1)
cv.imwrite("part1_A2.png", closing)

################### A3.PNG ###################
a3 = cv.imread('THE3-Images/A3.png', 0)
a3 = gammaCorrection(a3, 3)
thresh = cv.adaptiveThreshold(a3, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 141, 15)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (29, 29))
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

num, labels = cv.connectedComponents(closing)
print("The number of flying balloons in image A3 is", num - 1)
cv.imwrite("part1_A3.png", closing)

################### A4.PNG ###################
a4 = cv.imread('THE3-Images/A4.png', 0)
thresh_high = 10
thresh_low = 0
bool_img = (a4 <= thresh_high) & (a4 >= thresh_low)
binarized = np.uint8(bool_img * 255)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
opening = cv.morphologyEx(binarized, cv.MORPH_OPEN, kernel)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

num, labels = cv.connectedComponents(closing)
print("The number of flying balloons in image A4 is", num - 2)
cv.imwrite("part1_A4.png", closing)

################### A5.PNG ###################
a5 = cv.imread('THE3-Images/A5.png', 0)
thresh = cv.adaptiveThreshold(a5, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 85, 36)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

num, labels = cv.connectedComponents(opening)
print("The number of flying balloons in image A5 is", num - 1)
cv.imwrite("part1_A5.png", opening)



########################################### PART B ###########################################
def segmentation_function_meanshift(input_path):
    img = cv.imread(input_path)
    img = cv.GaussianBlur(img, (5, 5), sigmaX=1)
    flat_image = img.reshape((-1, 3))
    flat_image = np.float32(flat_image)
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    resized_image = resized_image.reshape((-1, 3))
    resized_image = np.float32(resized_image)
    bandwidth = estimate_bandwidth(resized_image, quantile=.07, n_samples=2000)
    ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
    ms.fit(flat_image)
    segments = np.unique(ms.labels_)
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(ms.labels_):
        total[label] += flat_image[i]
        count[label] += 1
    avg = np.uint8(total / count)
    res = avg[ms.labels_]
    result = res.reshape(img.shape)
    return result


def segmentation_function_ncut(input_path):
    img = imread(input_path)
    labels_slic = segmentation.slic(img, compactness=20, n_segments=1000)
    g = graph.rag_mean_color(img, labels_slic, mode='similarity')
    ncuts_labels = graph.cut_normalized(labels_slic, g)
    ncuts_result = color.label2rgb(ncuts_labels, img, kind='avg')
    ncut_image = cv.cvtColor(ncuts_result, cv.COLOR_BGR2RGB)
    return ncut_image


the3_B1output_meanshift = segmentation_function_meanshift(input_path=B1)
if not os.path.exists(output_path):
    os.makedirs(output_path)
cv.imwrite(output_path + '/the3_B1output_meanshift.png', the3_B1output_meanshift)

the3_B1output_ncut = segmentation_function_ncut(input_path=B1)
if not os.path.exists(output_path):
    os.makedirs(output_path)
cv.imwrite(output_path + '/the3_B1output_ncut.png', the3_B1output_ncut)
the3_B2output_meanshift = segmentation_function_meanshift(input_path=B2)
if not os.path.exists(output_path):
    os.makedirs(output_path)
cv.imwrite(output_path + '/the3_B2output_meanshift.png', the3_B2output_meanshift)

the3_B2output_ncut = segmentation_function_ncut(input_path=B2)
if not os.path.exists(output_path):
    os.makedirs(output_path)
cv.imwrite(output_path + '/the3_B2output_ncut.png', the3_B2output_ncut)

the3_B3output_meanshift = segmentation_function_meanshift(input_path=B3)
if not os.path.exists(output_path):
    os.makedirs(output_path)
cv.imwrite(output_path + '/the3_B3output_meanshift.png', the3_B3output_meanshift)

the3_B3output_ncut = segmentation_function_ncut(input_path=B3)
if not os.path.exists(output_path):
    os.makedirs(output_path)
cv.imwrite(output_path + '/the3_B3output_ncut.png', the3_B3output_ncut)
