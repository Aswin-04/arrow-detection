import cv2
import numpy as np
import time

# --- 1. CAPTURE IMAGE FROM WEBCAM ---
cam = cv2.VideoCapture(0)
cv2.namedWindow("Press 's' to capture")

print("Press 's' to capture a picture of the arrow.")
print("Press 'ESC' to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Press 's' to capture", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC key
        print("Exiting application.")
        break
    elif k % 256 == ord('s'):  # 's' key
        print("Image captured.")
        cv2.imwrite("captured_image.png", frame)
        break

cam.release()
cv2.destroyAllWindows()

# --- 2. LOAD IMAGES AND TEMPLATES ---
main_image = cv2.imread("captured_image.png", 0)
template_right = cv2.imread("right arrow.jpg", 0)
template_left = cv2.imread("left arrow.jpg", 0)

if main_image is None or template_right is None or template_left is None:
    print("Error: Could not load one or more images.")
    exit()

# Get dimensions of templates for drawing rectangle
w_right, h_right = template_right.shape[::-1]
w_left, h_left = template_left.shape[::-1]

# --- 3. PERFORM TEMPLATE MATCHING ---
res_right = cv2.matchTemplate(main_image, template_right, cv2.TM_CCOEFF_NORMED)
res_left = cv2.matchTemplate(main_image, template_left, cv2.TM_CCOEFF_NORMED)

_, max_val_r, _, max_loc_r = cv2.minMaxLoc(res_right)
_, max_val_l, _, max_loc_l = cv2.minMaxLoc(res_left)

# --- 4. CLASSIFY THE ARROW WITH A THRESHOLD ---
# A higher threshold is more strict. Adjust this value!
threshold = 0.75 

'''print(f"Right arrow match score: {max_val_r:.2f}")
print(f"Left arrow match score: {max_val_l:.2f}")

if max_val_r > threshold and max_val_r > max_val_l:
    print("The detected image is a RIGHT ARROW.")
    top_left = max_loc_r
    bottom_right = (top_left[0] + w_right, top_left[1] + h_right)
    cv2.rectangle(main_image, top_left, bottom_right, (255, 0, 0), 2)
    cv2.imshow("Detected Right Arrow", main_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif max_val_l > threshold and max_val_l > max_val_r:
    print("The detected image is a LEFT ARROW.")
    top_left = max_loc_l
    bottom_right = (top_left[0] + w_left, top_left[1] + h_left)
    cv2.rectangle(main_image, top_left, bottom_right, (255, 0, 0), 2)
    cv2.imshow("Detected Left Arrow", main_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No clear arrow detected in the image.")'''

if max_val_l>max_val_r:
    print("The given image is a left arrow")
else:
    print("The given image is a right arrow")