''' some image manipulation utility functions '''
import imutils
import cv2
import numpy as np


W = 124
H = 106
STEP = 5 #degree

def center_img(img):
    ''' translate the object mask to the center of the image plane '''
    center = get_center(img)
    ori_img = transform_img(img.copy(), 0, int(W/2 - center[1]), int(H/2 - center[0]))
    return ori_img, center - np.array([H/2, W/2])

def get_center(img):
    '''
    return the geometric center of an image blob
    '''
    if len(img.shape) == 3:
        img = img.copy()[:,:,0]

    (yidx, xidx) = np.where(img > 1)
    coords = np.array([[yidx[i], xidx[i]] for i in range(len(yidx))])
    center = coords.mean(axis=0)

    return center

def get_img_transform(from_img, to_img, symmetric=False):
    '''
    get rotation and translation required to move from_img to to_img
    '''
    from_center = get_center(from_img)
    to_center = get_center(to_img)
    ## get diff in translation
    diff_tran = to_center - from_center

    ## move both to center of the image
    from_img_ori = transform_img(from_img.copy(), 0, int(W/2 - from_center[1]), int(H/2 - from_center[0]))
    to_img_ori = transform_img(to_img.copy(), 0, int(W/2 - to_center[1]), int(H/2 - to_center[0]))

    ## rotate from_img in 360 degree to check how much it overlaps with to_img
    max_overlap = -10000
    best_w = 0
    best_w_list = []
    for i in range(int(180/STEP)):
        dw = -90 + i * STEP
        dummy_img = transform_img(from_img_ori.copy(), dw, 0, 0)
        num_overlap = count_overlap(dummy_img.copy(), to_img_ori.copy())
        if num_overlap > max_overlap:
            max_overlap = num_overlap
            best_w = dw
            if num_overlap > 0.95:
                best_w_list.append(dw)

    if max_overlap > 0.95:
        idx = np.argmin(np.abs(np.array(best_w_list)))
        return diff_tran, best_w_list[idx]
    else:
        return diff_tran, best_w

def generate_goal_img(img, w, x, y):
    ''' generate goal image in original image frame'''
    center = get_center(img)
    ## move it to the center
    img_ori = transform_img(img.copy(), 0, int(W/2 - center[1]), int(H/2 - center[0]))
    ## transform
    img_ = transform_img(img_ori.copy(), w, x, y)
    ## move it back
    img_f = transform_img(img_.copy(), 0, -int(W/2 - center[1]), -int(H/2 - center[0]))

    return img_f


def transform_img(img, w, x, y):
    '''
    rotate by w (degree) first
    then translate by x, y (pixel)
    '''
    img_rot = imutils.rotate(img.copy(), w)
    img_tran = imutils.translate(img_rot.copy(), x, y)
    return img_tran

def count_overlap(img1, img2):
    '''
    count number of overlapping pixels
    '''
    (y1, x1) = np.where(img1>1)
    (y2, x2) = np.where(img2>1)

    set1 = set(((y1[i], x1[i]) for i in range(len(y1))))
    set2 = set(((y2[i], x2[i]) for i in range(len(y2))))

    return 1.0 * len(set1.intersection(set2)) / len(y2)


''' Test Functions '''

def test_transform():
    img = cv2.imread('test.jpg')[:,:,0]
    w = 19
    x = 42
    y = 5
    center = get_center(img.copy())

    # centered image
    img_c = transform_img(img.copy(), 0, int(W/2-center[1]), int(H/2-center[0]))

    # transformed image
    img_t = transform_img(img_c.copy(), w, x, y)

    # offset image
    img_f = transform_img(img_t.copy(), 0, -int(W/2-center[1]), -int(H/2-center[0]))

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imshow('img', img_c)
    cv2.waitKey(0)
    cv2.imshow('img', img_t)
    cv2.waitKey(0)
    cv2.imshow('img', img_f)
    cv2.waitKey(0)

    ## estimate transformation
    print 'true transformation:'
    print y, x, w
    print 'estimated transformation:'
    print get_img_transform(img.copy(), img_f.copy())

#test_transform()














