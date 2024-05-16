# pose_detector.py
import cv2
import mediapipe as mp
import time



class PoseDetector:

    def __init__(self, mode: bool = False, up_body: bool = False, smooth: bool = True,
                 detection_con: float = 0.5, track_con: float = 0.5):

        self.mode = mode
        self.up_body = up_body
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        try:
            self.pose = self.mp_pose.Pose(static_image_mode=mode, model_complexity=1,
                                          smooth_landmarks=smooth, enable_segmentation=up_body,
                                          min_detection_confidence=detection_con,
                                          min_tracking_confidence=track_con)
        except TypeError as e:
            print(f"Error creating Pose object: {e}")

    def find_pose(self, img: cv2.Mat, draw: bool = True) -> cv2.Mat:

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 1, (255, 0, 0), cv2.FILLED)

        return lmlist


def main():
    cap = cv2.VideoCapture("PoseVideos/1.mp4")  # 0 is the default camera, you can change it if you have multiple cameras
    pTime = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break


if __name__ == "__main__":
    main()
