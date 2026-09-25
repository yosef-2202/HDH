import cv2
import mediapipe as mp
import time

class PoseEstimator:
    def __init__(self, video_path=None, camera_index=0, mode=False, model_complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        self.mpDraw = mp.solutions.drawing_utils
        self.pTime = 0
        self.results = None

    def process_frame(self):
        success, img = self.cap.read()
        if not success:
            return None

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (720, 1080)) 
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        cTime = time.time()
        fps = 1 / (cTime - self.pTime) if (cTime - self.pTime) != 0 else 0
        self.pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        return img

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_position(self, img, landmark_ids=None, draw=True):
        """
        Returns a list of (id, x, y) for specified landmark_ids.
        If landmark_ids is None, returns all landmarks.
        """
        lmList = []
        if self.results and self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                if (landmark_ids is None) or (id in landmark_ids):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        return lmList


def main():
    # To use a video file, pass its path as video_path
    # pose_estimator = PoseEstimator(video_path="PoseVideos/1.mp4")
    # To use webcam, leave video_path as None
    pose_estimator = PoseEstimator(video_path="PoseVideos/3.mp4")
    
    while True:
        img = pose_estimator.process_frame()
        if img is None:
            break
        
        positions = pose_estimator.get_position(img, landmark_ids=[0, 11, 12])  # Example: get nose and shoulders
        print(positions)
        
        cv2.imshow("Pose Estimation", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pose_estimator.release()

if __name__ == "__main__":
    main()