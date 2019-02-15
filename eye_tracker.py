import numpy as np
import cv2
import time

eye_cascade = cv2.CascadeClassifier(
    'haarcascade/haarcascade_eye_tree_eyeglasses.xml')


def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_eyes(img):
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(
        face_img, scaleFactor=1.2, minNeighbors=5)

    return eyes


def get_center_of_eyes(img):
    eyes_coords = detect_eyes(img)
    eyes = []

    for (x, y, w, h) in eyes_coords:
        center_x = int((x*2+w)/2)
        center_y = int((y*2+h)/2)
        eyes.append([center_x, center_y])

    if len(eyes) == 2:
        eyes = sorted(eyes, key=lambda x: x[0])

    return eyes


def get_center_of_eyes_coords(coords):
    eyes = []

    for (x, y, w, h) in coords:
        center_x = int((x*2+w)/2)
        center_y = int((y*2+h)/2)
        eyes.append([center_x, center_y])

    if len(eyes) == 2:
        eyes= sorted(eyes, key=lambda x: x[0])

    return eyes

def draw_circle_on_eyes(image,eyes):
    colors= [(255,0,0),(0,0,255)]

    for i,eye in enumerate(eyes):
        x= eye[0]
        y= eye[1]
        cv2.circle(img=image, center=(x,y), radius=10, color=colors[i], thickness=-1)


USE_CACHE= True
MAX_CACHE_COUNT= 5

def run_eye_color_mask():
    cap = cv2.VideoCapture(0) 
    cv2.namedWindow('eye_tracker',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('eye_tracker', 600,600)
    cached_eyes =[]
    cached_count=0

    while True:
        _, frame = cap.read(0) 
        eyes = get_center_of_eyes(frame)

        if len(eyes) != 2:
            print("wrong number of eyes(%d) have been detected" % len(eyes))
            if USE_CACHE & (MAX_CACHE_COUNT >= cached_count):
                print("using cache #%d"% cached_count)
                draw_circle_on_eyes(frame,cached_eyes)
                cached_count += 1

        else:
            print("detected eyes order at {0} {1}".format(eyes[0],eyes[1]))
            draw_circle_on_eyes(frame,eyes)
            if USE_CACHE:
                cached_eyes= eyes
                cached_count = 1


        cv2.imshow("eye_tracker", frame) 


        c = cv2.waitKey(1) 
        if c == 27: 
            break 

    cv2.destroyAllWindows()

def normalize_it(n1,n2,max_n,min_n):
    n= (n1+n2)/2
    return (n-min_n)/(max_n-min_n)


def shif_colors(image,eyes):
    canvas= np.zeros(image.shape,np.uint8)
    
    for eye in eyes:
        x= eye[0]
        y= eye[1]
        cv2.circle(img=canvas, center=(x,y), radius=15, color=(255,255,255), thickness=-1)
    
    max_x= canvas.shape[0]
    max_y= canvas.shape[1]
    distance= eyes[1][0]-eyes[0][0]

    multiplier= 1
    norm_x= normalize_it(eyes[0][0],eyes[1][0],max_x,0) * multiplier
    norm_y= normalize_it(eyes[0][1],eyes[1][1],max_y,0) * multiplier
    norm_z= normalize_it(distance,distance,300,50) * multiplier

    # image[canvas!=0] = (0,0,255)
    filter = np.uint8(np.multiply(canvas, [norm_x,norm_z,norm_y]))  
    image = image+filter

    return image

def write_text_on_image(img, text):
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img,text=text,org=(10,20), fontFace=font,fontScale= 0.5,color=(255,255,255),thickness=1,lineType=cv2.LINE_AA)

def run_eye_color_shift():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)/2
 
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter("output/output.avi",fourcc, fps, (frame_width,frame_height),True)

    while True:
        _, frame = cap.read(0) 
        eyes = get_center_of_eyes(frame)

        if len(eyes) != 2:
            message= "wrong number of eyes(%d) detected" % len(eyes)
            write_text_on_image(frame,message)
            print(message)
        else:
            message= "eyes detected: distance={0} loc={1},{2}".format(eyes[1][0]-eyes[0][0],eyes[0],eyes[1])
            write_text_on_image(frame,message)
            print(message)
            frame= shif_colors(frame,eyes)


        cv2.imshow("image", frame) 
        out.write(frame)

        c = cv2.waitKey(1) 
        if c == 27: 
            break 

    cv2.destroyAllWindows()


def detect_eyes_roi():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read(0) 
        eyes = detect_eyes(frame)
        eyes_roi =[]
        found= False
        
        if len(eyes) == 2:
            message= "Found eyes to track: distance={0} loc={1},{2}".format(eyes[1][0]-eyes[0][0],eyes[0],eyes[1])
            print(message)

            for (x,y,w,h) in eyes: 
                eyes_roi.append((y,x,y+h,x+w))

            frame= shif_colors(frame,eyes)
            found= True
            

        if found:
            time.sleep(1)
            return (frame,eyes)
    
    cv2.destroyAllWindows()

def ask_for_tracker():
    print("0- BOOSTING: ")
    print("1- MIL: ")
    print("2- KCF: ")
    print("3- TLD: ")
    print("4- MEDIANFLOW: ")
    choice = input("Please select your tracker: ")

    return choice

def get_tracker(choice):
    if choice == '0':
        tracker = cv2.TrackerBoosting_create()
    if choice == '1':
        tracker = cv2.TrackerMIL_create()
    if choice == '2':
        tracker = cv2.TrackerKCF_create()
    if choice == '3':
        tracker = cv2.TrackerTLD_create()
    if choice == '4':
        tracker = cv2.TrackerMedianFlow_create()

    return tracker

def use_tracker(firstframe,roi):
    tracker_number = ask_for_tracker()
    tracker = cv2.MultiTracker_create()
    
    tracker.add(get_tracker(tracker_number), firstframe, tuple(roi[0]))
    tracker.add(get_tracker(tracker_number), firstframe, tuple(roi[1]))
    
    tracker_name = str(get_tracker(tracker_number)).split()[0][1:]

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)


    while True:
        _, frame = cap.read()
        success, roi = tracker.update(frame)

        (x_1,y_1,w_1,h_1) = tuple(map(int,roi[0]))
        (x_2,y_2,w_2,h_2) = tuple(map(int,roi[1]))

        eyes= get_center_of_eyes_coords([(x_1,y_1,w_1,h_1),(x_2,y_2,w_2,h_2)])

        if success:
            message= "distance={0} loc={1},{2} tracker={3}".format(eyes[1][0]-eyes[0][0],eyes[0],eyes[1],tracker_name)
            write_text_on_image(frame,message)
            print(message)
            frame= shif_colors(frame,eyes)
        else :
            cv2.putText(frame, "image", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

        cv2.imshow("image", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27 : 
            break
            
    cap.release()
    cv2.destroyAllWindows()


firstframe, eyes_rois= detect_eyes_roi()
print(eyes_rois)
use_tracker(firstframe,eyes_rois)

# run_eye_color_shift()
