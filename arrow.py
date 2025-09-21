import cv2
import numpy as np

def get_arr_dir(contour_approx):
    x, y, w, h = cv2.boundingRect(contour_approx) #rect
    points = [tuple(p[0]) for p in contour_approx]

    center_x = x + w / 2 #hor
    center_y = y + h / 2 #ver

    max_distance = -1
    tip_of_arrow = None

    for p in points:
        distance = np.sqrt(((p[0] - center_x) ** 2) + ((p[1] - center_y) ** 2))
        if distance > max_distance:
            max_distance = distance
            tip_of_arrow = p

    if tip_of_arrow is None:
        return "UNKNOWN"

    
    dx = tip_of_arrow[0] - center_x #cen to hor
    dy = tip_of_arrow[1] - center_y #cen to ver

    if abs(dx) > abs(dy): #hor > ver
        if dx > 0:
            return "RIGHT"
        else:
            return "LEFT"
    else:
        if dy > 0:
            return "DOWN"
        else:
            return "UP"

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting.")
            break
        processed_image = frame.copy()

        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #outline
        for contour in contours:
            if cv2.contourArea(contour) < 500:  #500 pixel noice
                continue

            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.04 * perimeter #straight line
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 7:
                direction = get_arr_dir(approx)

                if direction in ["LEFT", "RIGHT"]:
                    cv2.drawContours(processed_image, [approx], -1, (0, 255, 0), 3)

                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(processed_image, direction, (cX - 40, cY),  #blue left
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                        print(f"Detected Arrow! Direction: {direction}")

        cv2.imshow("Arrow Detection (Press 'q' to quit)", processed_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()

