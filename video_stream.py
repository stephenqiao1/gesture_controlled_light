import cv2
import mediapipe as mp

def capture_video_stream():
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    while True:
        ret, frame = cap.read()
        
        flipped_frame = cv2.flip(frame, 1)
        
        if ret:
            cv2.imshow('Video Stream', flipped_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    capture_video_stream()