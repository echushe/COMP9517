##################################################
#
#              COMP9517 Project Stage 1
#              Chunnan Sheng
#              5100764
#
#######################################

# Please use Python 3.5 or higher version to run this program
# Please use OpenCV 3.3 Contrib or higher version to run this program

# There are four options to run this program

'''
Python3 ObjectTracker.py  <video_file_name> <thread_mode> <tracking_mode>
There are two thread modes: 0 as multi-thread mode, 1 as single thread mode.
There are two tracking modes: 0 as static tracking mode, 1 as dynamic tracking mode.

for example:

Python3 ObjectTracker.py  test_sample.mp4  1  0

'''
import cv2
import sys
import math
import threading
import numpy as np
import queue
import math


def draw_rects_on_image(img, rects):
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

def draw_contours_on_image(img, contours, index):
    index = index % 6;
    color = 0
    if index == 0:
        color = (0, 0, 255)
    elif index == 1:
        color = (255, 255, 0)
    elif index == 2:
        color = (0, 255, 0)
    elif index == 3:
        color = (0, 255, 255)
    elif index == 4:
        color = (0, 0, 255)
    else:
        color = (255, 0, 255)
    cv2.drawContours(img, contours, -1, color, 2)


def sub_images_via_rect(img, rects):
    sub_imgs = []
    for (x, y, w, h) in rects:
        sub_img = img[y:(y + h), x:(x + w)]
        print (x, y, w, h)
        sub_imgs.append(sub_img.copy())
    return sub_imgs


class XContour:

    def __init__(self, _contour):
        self.currentContour = []

        self.currentContour = _contour
        self.currentBoundingRect = cv2.boundingRect(self.currentContour)

        (x, y, w, h) = self.currentBoundingRect
        self.rectArea = w * h
        cx = (x + x + w) / 2;
        cy = (y +y + h) / 2;
        self.centerPosition = (cx, cy)
        
        self.dblCurrentDiagonalSize = math.sqrt(math.pow(w, 2) + math.pow(h, 2))
        self.dblCurrentAspectRatio = 1.0 * w / h;
        
        self.blnStillBeingTracked = True;
        self.blnCurrentMatchFoundOrNewBlob = True
        self.intNumOfConsecutiveFramesWithoutAMatch = 0



def get_contours_via_one_frame(frame):

    _frame = frame.copy()   
    # Convert the frames to gray color
    _frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    _frame = cv2.bilateralFilter(_frame, 11, 17, 17)
    # Blur
    #_frame = cv2.GaussianBlur(_frame, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    '''
    for i in range(2):
        _frame = cv2.dilate(_frame, kernel)
        _frame = cv2.dilate(_frame, kernel)
        _frame = cv2.erode(_frame, kernel)
    '''
    edged = cv2.Canny(_frame, 10, 250)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
 
    #finding_contours 
    (image, possible_contours, hierachy) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = []
    xcontours = []
    for contour in possible_contours:
        convexHull = cv2.convexHull(contour);
        xc = XContour(convexHull);

        (x, y, w, h) = xc.currentBoundingRect;
        if (xc.rectArea > 200 and
            xc.dblCurrentAspectRatio > 0.2 and
            xc.dblCurrentAspectRatio < 4.0 and
            w > 30 and
            h > 30 and
            xc.dblCurrentDiagonalSize > 30.0 and
            (cv2.contourArea(xc.currentContour) * 1.0 / xc.rectArea) > 0.50):
            contours.append(convexHull);
            xcontours.append(xc)

    rects = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        rects.append(rect);

    return (rects, contours, xcontours)



def get_contours_via_two_frames(frame_1, frame_2):
    l_frame_1 = frame_1.copy()
    l_frame_2 = frame_2.copy()
    
    # Convert the frames to gray color
    l_frame_1 = cv2.cvtColor(l_frame_1, cv2.COLOR_BGR2GRAY)
    l_frame_2 = cv2.cvtColor(l_frame_2, cv2.COLOR_BGR2GRAY)

    # Blur
    l_frame_1 = cv2.GaussianBlur(l_frame_1, (5, 5), 0)
    l_frame_2 = cv2.GaussianBlur(l_frame_2, (5, 5), 0)

    #
    difference = cv2.absdiff(l_frame_1, l_frame_2);
    (retval, threshold) = cv2.threshold(difference, 10, 255.0, cv2.THRESH_BINARY)

    structuringElement5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
 
    for i in range(2):
        threshold = cv2.dilate(threshold, structuringElement5x5)
        threshold = cv2.dilate(threshold, structuringElement5x5)
        threshold = cv2.erode(threshold, structuringElement5x5)

    (image, possible_contours, hierachy) = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = []
    xcontours = []
    for contour in possible_contours:
        convexHull = cv2.convexHull(contour);
        xc = XContour(convexHull);

        (x, y, w, h) = xc.currentBoundingRect;
        if (xc.rectArea > 200 and
            xc.dblCurrentAspectRatio > 0.2 and
            xc.dblCurrentAspectRatio < 4.0 and
            w > 30 and
            h > 30 and
            xc.dblCurrentDiagonalSize > 30.0 and
            (cv2.contourArea(xc.currentContour) * 1.0 / xc.rectArea) > 0.50):
            contours.append(convexHull);
            xcontours.append(xc)

    rects = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        rects.append(rect);

    return (rects, contours, xcontours)



def drawPoints(img, points):
    for i in range(len(points) - 1):
        (x1, y1) = points[i]
        (x2, y2) = points[i + 1]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255 ,255), 2)



def match_2_frames(kp_des1, img1, img2, img2_mask, r_contours, xcontours, surf, routes, index): 
    img2_copy = img2.copy()
    kp_des2 = surf.detectAndCompute(img2_copy, img2_mask)
    return draw_matches(kp_des1, kp_des2, img1, img2_copy, r_contours, xcontours, routes, index)


def draw_matches(kp_des1, kp_des2, img1, o_img2, r_contours, r_xcts, routes, index):
    # img2_for_match = cv2.bitwise_and(img2_copy, img2_copy, img2_mask)
    kp1, des1 = kp_des1
    kp2, des2 = kp_des2
    img2 = o_img2.copy()

    points = []
    if None != routes:
        while not routes.empty():
            points.append(routes.get())
        for point in points:
            routes.put(point)
        drawPoints(img2, points)

    index = index % 6;
    color = 0
    if index == 0:
        color = (0, 0, 255)
    elif index == 1:
        color = (255, 255, 0)
    elif index == 2:
        color = (0, 255, 0)
    elif index == 3:
        color = (0, 255, 255)
    elif index == 4:
        color = (0, 0, 255)
    else:
        color = (255, 0, 255)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1, des2, k = 2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    m_points = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1,0]
            m_points.append(kp2[n.trainIdx].pt)

    x0 = 0.0
    y0 = 0.0
    for point in m_points:
        (x, y) = point
        x0 += x
        y0 += y
    if (len(m_points) > 0):
        x0 /= len(m_points)
        y0 /= len(m_points)

    min_dist = 10000
    contour_to_draw = []
    r_xct = None
    if not None == r_xcts:
        for i in range(len(r_xcts)):
            (x, y) = r_xcts[i].centerPosition
            dist = math.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0))
            if min_dist > dist:
                min_dist = dist
                contour_to_draw = r_contours[i]
                r_xct = r_xcts[i]


    if len(contour_to_draw) > 0 and len(matches) > 5:
        # print("================== draw R contours ===================")
        cv2.drawContours(img2, [contour_to_draw], -1, color, 2)
        (x0, y0) = r_xct.centerPosition

    (c1, c2, c3) = color
    color_ = (255 - c1, 255 - c2, 255 - c3)
    draw_params = dict(matchColor = color,
                       singlePointColor = (0, 0, 0),
                       matchesMask = matchesMask,
                       flags = 2)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)    

    return (img3, (x0, y0))




class MatchingThread(threading.Thread):
     def __init__(self, kp_des, sub_image, frames, surf, obj_id, queue, detect_mode):
         super(MatchingThread, self).__init__()
         self.kp_des = kp_des
         self.sub_image = sub_image
         self.surf = surf
         self.obj_id = obj_id
         self.frames = frames
         self.queue = queue
         self.routes = []
         self.detect_mode = detect_mode
         # for frame in frames:
         #    self.frames.append(frame.copy());

     def run(self):
         print ('Thread: ' + str(self.obj_id))
         routes = queue.Queue()
         for i in range(len(self.frames) - 1):
             if (routes.qsize() > 30):
                 routes.get()
             frame1 = self.frames[i]
             frame2 = self.frames[i + 1]
             if 0 == detect_mode:
                 (rects, contours, xcontours) = get_contours_via_one_frame(frame1)
             else:
                 (rects, contours, xcontours) = get_contours_via_two_frames(frame1, frame2)

             #(rects, contours, xcontours) = get_contours_via_two_frames(frame1, frame2)

             h = np.size(frame1, 0)
             w = np.size(frame2, 1)
             frame_1_mask = np.zeros((h, w, 1), np.uint8) # cv2.(imageSize, CV_8UC3, SCALAR_BLACK);

             cv2.drawContours(frame_1_mask, contours, -1, (255, 255, 255), -1)

             try:
                 (m_img, point) = match_2_frames(self.kp_des, self.sub_image, frame1, frame_1_mask, contours, xcontours, self.surf, routes, self.obj_id)
                 self.queue.put(m_img)

                 (x, y) = point
                 if x > 0 and y > 0:
                    routes.put(point)
             except cv2.error:
                 print('error')

             # print ('image ' + str(self.obj_id))



def split_and_match(video, detect_mode):
    # obj_id = int(sys.argv[2])
 
    # Exit if video not opened.
    if not video.isOpened():
        print ('Could not open video')
        sys.exit()

    l_kp_des = []
    threads = []
    video_frames = [];
    queues = [];
    # represents the addition of an item to a resource
    event = threading.Event()

    frame_1 = 0
    frame_2 = 0

    surf = cv2.xfeatures2d.SURF_create(500)
    #surf = cv2.ORB_create()

    index = 0
    while (True):
        print(index)
        # Read first frame.
        (ok, frame_1) = video.read()
        if not ok:
            print ('Cannot read video file')
            sys.exit()

        (ok, frame_2) = video.read()
        if not ok:
            print ('Cannot read video file')
            sys.exit()

        if 0 == detect_mode:
            (l_rects, l_contours, l_xcontours) = get_contours_via_one_frame(frame_1)
        else:
            (l_rects, l_contours, l_xcontours) = get_contours_via_two_frames(frame_1, frame_2)

        if len(l_rects) > 2:

            while True:
                # Read a new frame
                (ok, frame) = video.read()
                if not ok:
                    break
                video_frames.append(frame);

            l_sub_images = sub_images_via_rect(frame_1, l_rects)

            for x in range(len(l_sub_images)):
                queues.append(queue.Queue())

            for x in range(len(l_sub_images)):
                (kp, des) = surf.detectAndCompute(l_sub_images[x], None)
                l_kp_des.append((kp, des))
                threads.append(MatchingThread(l_kp_des[x], l_sub_images[x], video_frames, surf, x, queues[x], detect_mode))

            # draw_rects_on_image(frame_1, l_rects)
            draw_contours_on_image(frame_1, l_contours, 0)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', frame_1)
            break
        index += 1

    for x in range(len(l_kp_des)):
         threads[x].start() # This actually causes the thread to run

    # consumer thread
    
    while True:
        # event.wait() # sleep until item becomes available
        for y in range(len(queues)):
            if not queues[y].empty():
                new_frame = queues[y].get()
                cv2.namedWindow('image show ' + str(y), cv2.WINDOW_NORMAL)
                cv2.imshow('image show ' + str(y), new_frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break     

    for y in range(len(l_kp_des)):
         threads[y].join()  # This waits until the thread has completed


def entire_match(video, detect_mode):
    # Exit if video not opened.
    if not video.isOpened():
        print ('Could not open video')
        sys.exit()

    l_kp_des_list = []
    frame_1 = 0
    frame_2 = 0
    frame_3 = 0
    frame_4 = 0

    surf = cv2.xfeatures2d.SURF_create(500)
    #surf = cv2.ORB_create()
    index = 0
    while (True):
        print(index)
        # Read first frame.
        (ok, frame_1) = video.read()
        if not ok:
            print ('Cannot read video file -')
            sys.exit()

        (ok, frame_2) = video.read()
        if not ok:
            print ('Cannot read video file --')
            sys.exit()

        if 0 == detect_mode:
            (l_rects, l_contours, l_xcontours) = get_contours_via_one_frame(frame_1)
        else:
            (l_rects, l_contours, l_xcontours) = get_contours_via_two_frames(frame_1, frame_2)

        if len(l_rects) > 2:

            h = np.size(frame_1, 0)
            w = np.size(frame_1, 1)

            for ct in l_contours:
                mask = np.zeros((h, w, 1), np.uint8)
                cv2.drawContours(mask, [ct], -1, (255, 255, 255), -1)
                l_kp_des = surf.detectAndCompute(frame_1, mask)
                l_kp_des_list.append(l_kp_des)          

            # draw_rects_on_image(frame_1, l_rects)
            for i in range(len(l_contours)):
                draw_contours_on_image(frame_1, [l_contours[i]], i)
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.imshow('image', frame_1)
            # cv2.imshow('Mask', frame_1_mask)
            break
        index += 1
    
    while True:
        (ok, frame_3) = video.read()
        if not ok:
            print ('Cannot read video file ---')
            sys.exit()

        (ok, frame_4) = video.read()
        if not ok:
            print ('Cannot read video file ----')
            sys.exit()

        f3_h = np.size(frame_3, 0)
        f3_w = np.size(frame_3, 1)
        frame_3_mask = np.zeros((f3_h, f3_w, 1), np.uint8)

        if 0 == detect_mode:
            (r_rects, r_contours, r_xcontours) = get_contours_via_one_frame(frame_3)
        else:
            (r_rects, r_contours, r_xcontours) = get_contours_via_two_frames(frame_3, frame_4)

        cv2.drawContours(frame_3_mask, r_contours, -1, (255, 255, 255), -1)
        
        r_kp_des = surf.detectAndCompute(frame_3, frame_3_mask)      
        
        try:
            cv2.namedWindow('image show ', cv2.WINDOW_NORMAL)
            f1_h = np.size(frame_1, 0)
            f1_w = np.size(frame_1, 1)
            img_to_show = np.zeros((f3_h, f1_w + f3_w, 3), np.uint8)
            frame_1_mock = np.zeros((f1_h, f1_w, 3), np.uint8)
            frame_3_mock = np.zeros((f3_h, f3_w, 3), np.uint8)
            
            for i in range(0, len(l_kp_des_list)):
                l_img_to_show = None
                                                    
                (l_img_to_show, point) = draw_matches(l_kp_des_list[i], r_kp_des, frame_1_mock, frame_3_mock, r_contours, r_xcontours, None, i)
                img_to_show = cv2.bitwise_or(img_to_show, l_img_to_show)
                if i == len(l_kp_des_list) - 1:
                    background = np.concatenate((frame_1, frame_3), axis=1)
                    img_to_show = cv2.bitwise_or(img_to_show, background)
                
           
            cv2.imshow('image show ', img_to_show)        
            # print ('image show')
        except cv2.error: 
            print('Error!')


        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break     



if __name__ == '__main__' :
 
    # Read video
    video = cv2.VideoCapture(sys.argv[1])

    option = int(sys.argv[2])
    detect_mode = int(sys.argv[3])

    if option == 0:
        split_and_match(video, detect_mode)
    elif option == 1:
        entire_match(video, detect_mode)


        