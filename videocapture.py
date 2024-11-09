import os
from mtcnn import MTCNN
import cv2
import tensorflow as tf

# Initialize the MTCNN face detector
detector = MTCNN()

# Open a connection to the default camera (0)
cap = cv2.VideoCapture(0)

face_folder = 'output_faces'



# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0 #start frame counter

# Loop over frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break


    frame_count += 1
    text = f"{frame_count}"
    cv2.putText(frame, text, (20,30),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)

    # Convert the frame to RGB (MTCNN expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    results = detector.detect_faces(rgb_frame)

    # Draw bounding boxes and landmarks for each detected face
    for result in results:
        x, y, width, height = result['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

        # Face Croping from frame
        frame_crop = frame[y:y + height, x:x + width]
        frame_filename = os.path.join(face_folder, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame_crop)


        # Draw facial landmarks
        for key, point in result['keypoints'].items():
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

    # Display the frame with detections
    cv2.imshow("MTCNN Detected Faces", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
