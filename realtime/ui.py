import cv2

class UI:
    def __init__(self):
        self.score = 0

    def update(self, frame, score):
        cv2.putText(frame, f"Prediction: {score}%", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Exercise 5", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.imshow('output', frame)
        self.score = score

    def should_quit():
        return cv2.waitKey(1) & 0xFF == ord('q')