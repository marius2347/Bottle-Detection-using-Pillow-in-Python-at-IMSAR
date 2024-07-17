import cv2
import numpy as np
import os

# Set the paths
image_folder = './calib_images'
output_folder = './output_images'
keypoints_file = './output_images/keypoints.txt'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load images from the folder
image_list = []
image_names = []

for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')): 
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            image_list.append(img)
            image_names.append(filename)

# Initialize webcam
cap = cv2.VideoCapture(4)

with open(keypoints_file, 'w') as kf:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        for img, name in zip(image_list, image_names):
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(frame_gray, img_gray, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(result >= threshold)

            h, w = img_gray.shape
            for pt in zip(*loc[::-1]):
                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Shi-Tomasi corner detection
                corners = cv2.goodFeaturesToTrack(frame_gray, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3)

                if corners is not None:
                    for corner in corners:
                        x, y = corner.ravel()
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        kf.write(f"{name}, x: {int(x)}, y: {int(y)}\n")

                output_image_path = os.path.join(output_folder, f"detected_{name}")
                cv2.imwrite(output_image_path, frame)

        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
