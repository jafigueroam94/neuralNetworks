import requests
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import image_to_string

x1=0
y1=0
x2=0
y2=0
url = "http://192.168.1.125:8080/"
linterna = False
font = cv2.FONT_HERSHEY_SIMPLEX


drawing = False
point1 = ()
point2 = ()

def setLinternaToogle(linterna):
    accion = "disabletorch" if linterna else "enabletorch"
    requests.get(url + accion)
    return not linterna
    
def setText(img, text):
    # cv2.putText(img, text) 
    cv2.putText(img, text,(50, 50), font, 1, (200,0,0), 3, cv2.LINE_AA)   

def getText(img):
    # Read image with opencv
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    result = pytesseract.image_to_string(img)

    return result


def mouse_drawing(event, x, y, flags, params):
    global point1, point2, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing is False:
            drawing = True
            point1 = (x, y)
        else:
            drawing = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            point2 = (x, y)


cap = cv2.VideoCapture(0)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)


while True:
    responseImg = requests.get(url + "shot.jpg")
    imageArray = np.array(bytearray(responseImg.content), dtype=np.uint8)
    frame = cv2.imdecode(imageArray, -1)

    if point1 and point2:
        cv2.rectangle(frame, point1, point2, (0, 255, 0), 6)
    
    x1 = int(frame.shape[1]/2 - (640/2))
    y1 = int(frame.shape[0]/2 - (400/2))
    x2 = int(frame.shape[1]/2 + (640/2))
    y2 = int(frame.shape[0]/2 + (400/2))
    cv2.rectangle(frame, ((x1, y1), (x2, y2)), (0, 255, 0), 6)
    setText(frame, getText(frame))
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == 8:
        linterna = setLinternaToogle(linterna)
        requests.get(url + "settings/motion_detect?set=on")
        requests.get(url + "settings/motion_display?set=on")
    if key == 32:
        clone = frame[point1[1]: point2[1], point1[0]: point2[0]]
        # cv2.imwrite('./assets/recorte.png',clone)
        print(getText(clone))

            

cap.release()
cv2.destroyAllWindows()