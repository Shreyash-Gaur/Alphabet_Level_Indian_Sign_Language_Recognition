import cv2
import os
import time
import uuid

# Make directories
IMAGES_PATH = 'Collected_Images'
labels = ['M']  # Labels 
number_imgs = 60

for label in labels:
    os.makedirs(os.path.join(IMAGES_PATH, label), exist_ok=True)

# Start image collection
for label in labels:
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(1)
    
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        
        # Generate unique filename based on current time
        current_time = time.strftime("%Y%m%d-%H%M%S")
        imgname = os.path.join(IMAGES_PATH, label, '{}_{}.jpg'.format(label, current_time))
        
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(100)  # Add a delay to display the frame
        time.sleep(1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
