import sys
import cv2
from datetime import datetime

# interval to save images (seconds)
kSaveImageDeltaTime = 1

def detect_bottle(image):
    # load the pre-trained Haar Cascade classifier for bottle detection
    cascade_path = 'haarcascade_upperbody.xml' 
    bottle_cascade = cv2.CascadeClassifier(cascade_path)
    # convert bgr to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect bottles in the image
    bottles = bottle_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    return bottles

if __name__ == "__main__":
    camera_num = 4
    if len(sys.argv) == 2:
        camera_num = int(sys.argv[1])
    print('Opening camera: ', camera_num)

    # initialize webcam
    webcam = cv2.VideoCapture(camera_num)
    if not webcam.isOpened():
        print(f"Error: Could not open camera {camera_num}")
        sys.exit(1)

    last_save_time = 0

    while True:
        current_time = datetime.now().timestamp()
        elapsed_time_since_last_save = current_time - last_save_time
        
        # capture frame from webcam
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to capture image from camera")
            break

        # detect bottles in the frame
        bottles = detect_bottle(frame)

        if len(bottles) > 0:
            print(f"Found {len(bottles)} bottle(s)")

            # save image if elapsed time is greater than kSaveImageDeltaTime
            if elapsed_time_since_last_save > kSaveImageDeltaTime:
                filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.bmp'
                image_path = "./calib_images/" + filename
                print("Saving " + filename)
                cv2.imwrite(image_path, frame)
                last_save_time = current_time

            # draw rectangles around the detected bottles
            for (x, y, w, h) in bottles:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('camera', frame)

        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
    
    # release webcam and close all OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()
