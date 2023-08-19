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
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Create a gesture recognizer class with the live stream mode
class gesture_and_result:
    def __init__(self):
        self.result = GestureRecognizerResult
        self.gesture = GestureRecognizer
        self.createGesture()
    
    def createGesture(self):
        # callback function
        def update_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result
            
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=update_result,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        
        # initialize gesture
        self.gesture = self.gesture.create_from_options(options)
        
    def recognize_async(self, frame):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect gestures
        self.gesture.recognize_async(image=mp_image, timestamp_ms=int(time.time() * 1000))
        
    def close(self):
        self.gesture.close()

class landmarker_and_result:
    def __init__(self):
        self.result = HandLandmarkerResult
        self.landmarker = HandLandmarker
        self.createLandmarker()

    def createLandmarker(self):
        # callback function
        def update_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=update_result,
            num_hands=2
        )

        # initialize landmarker
        self.landmarker = self.landmarker.create_from_options(options)

    def detect_async(self, frame):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect landmarks
        self.landmarker.detect_async(
            image=mp_image, timestamp_ms=int(time.time() * 1000)
        )

    def close(self):
        # close landmarker
        self.landmarker.close()
        

def create_convex_hull(height, width, hand_landmarks, annotated_image, mask):
    # Convert normalized landmarks to pixel coordinates
    pixel_coords = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_landmarks
    ]

    # Create a convex hull around the hand
    hull = cv2.convexHull(np.array(pixel_coords), returnPoints=False)

    # Draw the convex hull on the mask
    cv2.drawContours(
        mask, [np.array(pixel_coords)], 0, (255), -1
    )  # Fill the hand region with white

    # Find convexity defects
    defects = cv2.convexityDefects(np.array(pixel_coords), hull)

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(pixel_coords[s])
            end = tuple(pixel_coords[e])
            far = tuple(pixel_coords[f])

            # Draw the defects on the image
            cv2.line(annotated_image, start, end, [0, 255, 0], 2)
            cv2.circle(annotated_image, far, 5, [0, 0, 255], -1)
            
def display_gesture_on_image(rgb_image, gesture_result: GestureRecognizerResult):
    try:
        if gesture_result.gestures == []:
            return rgb_image
        else:
            gesture_list = gesture_result.gestures
            labeled_image = np.copy(rgb_image)
            
            for idx in range(len(gesture_list)):
                gestures = gesture_list[idx]
                
                 # Display the recognized gesture on the image
                cv2.putText(labeled_image, f"Gesture: {gestures[0].category_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return labeled_image
    except:
        return rgb_image


def draw_landmarks_on_image(rgb_image, detection_result: HandLandmarkerResult):
    try:
        height, width, _ = rgb_image.shape
        mask = np.zeros(
            (height, width), dtype=np.uint8
        )  # Create a black image for the mask

        if detection_result.hand_landmarks == []:
            return rgb_image, mask
        else:
            hand_landmarks_list = detection_result.hand_landmarks
            annotated_image = np.copy(rgb_image)

            # Loop through the detected hands to visualize
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                
                create_convex_hull(height, width, hand_landmarks, annotated_image, mask)

                # Draw the hand landmarks
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in hand_landmarks
                    ]
                )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )
        return annotated_image, mask
    except:
        return rgb_image, mask


def capture_video_stream():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # create landmarker
    hand_landmarker = landmarker_and_result()
    
    # create gestures
    gestures = gesture_and_result()

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        hand_landmarker.detect_async(frame)
        gestures.recognize_async(frame)
        print("Landmarker: ", hand_landmarker.result)
        print("Gestures: ", gestures.result)

        # draw landmarks on frame and get the mask
        frame, mask = draw_landmarks_on_image(frame, hand_landmarker.result)
        
        # display gesture labels
        frame = display_gesture_on_image(frame, gestures.result)

        cv2.imshow("Video Stream", frame)
        cv2.imshow("Hand Mask with Defects", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_video_stream()
