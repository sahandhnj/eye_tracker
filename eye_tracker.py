import numpy as np
import cv2 
import time

eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye_tree_eyeglasses.xml')

def detect_eyes(img):
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=5) 
        
    return eyes

def get_center_of_eyes(img):
    eyes_coords = detect_eyes(img)
    eyes=[]

    for (x,y,w,h) in eyes_coords:
        center_x= int((x*2+w)/2)
        center_y= int((y*2+h)/2)
        eyes.append([center_x,center_y])

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
    
    for i,eye in enumerate(eyes):
        x= eye[0]
        y= eye[1]
        cv2.circle(img=canvas, center=(x,y), radius=15, color=(255,255,255), thickness=-1)
    
    max_x= canvas.shape[0]
    max_y= canvas.shape[1]
    distance= eyes[1][0]-eyes[0][0]

    multiplier= 1.5
    norm_x= normalize_it(eyes[0][0],eyes[1][0],max_x,0) * multiplier
    norm_y= normalize_it(eyes[0][1],eyes[1][1],max_y,0) * multiplier
    norm_z= normalize_it(distance,distance,300,50) * multiplier

    # image[canvas!=0] = (0,0,255)
    filter = np.uint8(np.multiply(canvas, [norm_x,norm_y,norm_z]))  
    image = image+filter

    return image

def run_eye_color_shift():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)

    while True:
        _, frame = cap.read(0) 
        eyes = get_center_of_eyes(frame)

        if len(eyes) != 2:
            print("wrong number of eyes(%d) have been detected" % len(eyes))
        else:
            print("detected eyes distance={0} at {1} {2}".format(eyes[1][0]-eyes[0][0],eyes[0],eyes[1]))
            frame= shif_colors(frame,eyes)


        cv2.imshow("image", frame) 


        c = cv2.waitKey(1) 
        if c == 27: 
            break 

    cv2.destroyAllWindows()

run_eye_color_shift()



