import cv2
from pathlib import Path
import numpy as np

def user_inputs():
    rgb_wanted = input("Please input what rgb you want in this format r g b : ")
    rgb_list = list(map(int, rgb_wanted.split()))
    rgb_color = np.uint8([[rgb_list]])
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
    hsv_value = hsv_color[0][0]
    global filter_type
    if(hsv_value[0] < 3): filter_type = 2
    elif(hsv_value[0] > 176) : filter_type = 3
    else: filter_type = 1

    return hsv_value

def HSV_Conversion(image_to_convert, hsv_value, filter_type):

    IMG_DIR = Path(__file__).resolve().parent / str(image_to_convert)#this way of accessing images is kinda temp
    image = cv2.imread(str(IMG_DIR))

    if image is None: #error if cannot find image path
        print("Error: bad path to image in HSV conversion.")
        return
    
    low_res_img = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA) #makes image lower quality so can run faster
    hsv = cv2.cvtColor(low_res_img, cv2.COLOR_BGR2HSV) #converts to HSV

    target_hue = hsv_value[0]
    match (filter_type):
        case 1: #if everything is in range
            lower = np.array([target_hue - 15, 100, 100])
            upper = np.array([target_hue + 15, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

        case 2: #if it is out of range in the lower zone
            lowestlower = np.array([target_hue+176, 100, 100])
            lowestupper = np.array([179, 255, 255])
            upperlower = np.array([0, 100, 100])
            upperupper = np.array([target_hue+3,255,255])
            lowermask = cv2.inRange(hsv, lowestlower, lowestupper)
            uppermask = cv2.inRange(hsv, upperlower, upperupper)
            mask = cv2.bitwise_or(lowermask, uppermask)

        case 3: #if it is out of range in the upper zone
            lowestlower = np.array([target_hue-3, 100, 100])
            lowestupper = np.array([179, 255, 255])
            upperlower = np.array([0, 100, 100])
            upperupper = np.array([target_hue-176,255,255])
            lowermask = cv2.inRange(hsv, lowestlower, lowestupper)
            uppermask = cv2.inRange(hsv, upperlower, upperupper)
            mask = cv2.bitwise_or(lowermask, uppermask)

    #post processing on maks to get rid of small noise and fill in holes in chunks, is a little over ambitious
    """
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(pre_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    """
    
    mask = noise_filter(mask, method="gaussion")
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            x, y, w, h = cv2.boundingRect(largest)

            cv2.rectangle(low_res_img, (x, y), (x+w, y+h), (0,255,0), 2)
            
            zoomed = digital_zoom(low_res_img, x, y, w, h, zoom_factor=2.0)
            cv2.imshow("zoomed", zoomed)
    #disp stuff
    cv2.imshow("test",low_res_img)
    cv2.imshow("mymask :)",mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(x,y) #returns x and y cords of the centroid program id'd

def noise_filter(mask, method="gaussion"):
    if method == "gaussion":
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        _, cleaned = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    elif method == "median":
        cleaned = cv2.medianBlur(mask, 5)
    
    return cleaned

def digital_zoom(image, x, y, w, h, zoom_factor=2.0, padding=30):
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    
    cropped = image[y1:y2, x1:x2]
    
    new_w = int(cropped.shape[1] * zoom_factor)
    new_h = int(cropped.shape[0] * zoom_factor)
    zoomed = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed
    

hsv_value = user_inputs()
print(hsv_value)
print(HSV_Conversion("feetball.jpg", hsv_value, filter_type))
