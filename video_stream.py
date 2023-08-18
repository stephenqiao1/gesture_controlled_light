import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
    
class landmarker_and_result():
    def __init__(self):
        self.result = HandLandmarkerResult
        self.landmarker = HandLandmarker
        self.createLandmarker()    
    
    def createLandmarker(self):
    # callback function
        def update_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=update_result,
            num_hands=2,
            min_hand_detection_confidence=0.3, # lower than value to get predictions more often
            min_hand_presence_confidence=0.3, # lower than value to get predictions more often
            min_tracking_confidence=0.3, # lower than value to get predictions more often
        )
    
        # initialize landmarker
        self.landmarker = self.landmarker.create_from_options(options)
        
    def detect_async(self, frame):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect landmarks
        self.landmarker.detect_async(image=mp_image, timestamp_ms=int(time.time() * 1000))
        
    def close(self):
        # close landmarker
        self.landmarker.close()
        
def draw_landmarks_on_image(rgb_image, detection_result: HandLandmarkerResult):
    try:
        if detection_result.hand_landmarks == []:
            return rgb_image
        else:
            hand_landmarks_list = detection_result.hand_landmarks
            annotated_image = np.copy(rgb_image)
            
            # Loop through the detected hands to visualize
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                
                # Draw the hand landmarks
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
        return annotated_image
    except:
        return rgb_image
            
            
    
def capture_video_stream():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # create landmarker
    hand_landmarker = landmarker_and_result()
    
    while True:
        ret, frame = cap.read()
        
        frame = cv2.flip(frame, 1)
        
        hand_landmarker.detect_async(frame)
        print(hand_landmarker.result)
        
        # draw landmarks on frame
        frame = draw_landmarks_on_image(frame, hand_landmarker.result)
        
        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    capture_video_stream()