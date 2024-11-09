from mtcnn import MTCNN
import cv2
import tensorflow as tf

# Initialize the MTCNN face detector
detector = MTCNN()

# Load the image
image = cv2.imread("images/peopleB.jpg")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces
results = detector.detect_faces(rgb_image)

count = 0

for result in results:
    x, y, width, height = result['box']
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
    count += 1

    # Draw facial landmarks
    for key, point in result['keypoints'].items():
        cv2.circle(image, point, 2, (0, 255, 0), -1)

cv2.imshow("MTCNN Detected Faces", image)
cv2.imwrite("output/test2_detected.jpg", image)

print(count)

cv2.waitKey(0)
cv2.destroyAllWindows()
