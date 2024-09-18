# from tensorflow.python.keras.models import load_model
import mediapipe as mp
import numpy as np
import asyncio
import cv2
import tensorflow as tf

from preproccing_mediapipe_kinctv2 import mediapipe_to_kinect_v2, Keypoint

from ui import UI
from shared.data_processing import Data_Loader, Test_Data_Loader
from model.architectures.stgcn_lstm.gcn_layer import GCNLayer


class RealTimeApp:
    def __init__(self, exercise_folder, trained_model_path, ui,video_path=None):
        self.exercise_folder = exercise_folder
        self.trained_model_path = trained_model_path
        self.video_path = video_path
        self.ui = ui

        self.data_loader = Data_Loader(self.exercise_folder)
        self.model = tf.keras.models.load_model(self.trained_model_path, custom_objects={"GCNLayer": GCNLayer})
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def preprocess_frame(self,frame_list):
        frame_list_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frame_list]
        frame_list_results = [self.pose.process(frame_rgb) for frame_rgb in frame_list_rgb]
        frame_list_landmarks = [self.extract_landmarks(frame_results) for frame_results in frame_list_results]
        frame_list_vkv2 = [mediapipe_to_kinect_v2(landmarks) for landmarks in frame_list_landmarks]
        frame_list_vkv2_array = [np.array(vkv2) for vkv2 in frame_list_vkv2]
        frame_list_vkv2_3d = [vkv2_array[:, :3] for vkv2_array in frame_list_vkv2_array]
        frame_list_result = [np.array([coord for point in vkv2_3d for coord in point]) for vkv2_3d in frame_list_vkv2_3d]
        frame_list_result_array = np.array(frame_list_result)
        print("frame_list_result_array.shape: ", frame_list_result_array.shape)
        return Test_Data_Loader(frame_list_result_array)

    def extract_landmarks(self, frame_results):
        if frame_results.pose_landmarks:
            landmarks = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in frame_results.pose_landmarks.landmark]
            return landmarks
        else:
            return []

    def predict_batches(self, landmarks):
        predictions = [] 
        print("From here")
        for i in range(landmarks.scaled_x.shape[0]):
            prediction = self.model.predict(landmarks.scaled_x[i].reshape(1,landmarks.scaled_x[i].shape[0],landmarks.scaled_x[i].shape[1],landmarks.scaled_x[i].shape[2]))
            predictions.append(prediction[0,0])
        return predictions

    def run(self, batch_size=100): 
        cap = cv2.VideoCapture(0) if self.video_path is None else cv2.VideoCapture(self.video_path)

        frames_collector = []
        predictions = []
        score = 0


        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_collector.append(frame)

            if (len(frames_collector) >= batch_size):
                landmarks = self.preprocess_frame(frames_collector)
                predictions = self.predict_batches(landmarks) if landmarks is not None else None
                for i in range(len(predictions)):
                    print(f"Prediction {i}: {predictions[i]}")
                score = round(predictions[-1] * 100) if predictions[-1] > 0 else 0

            self.ui.update(frame, score)


            # if self.ui.should_quit():
            #     break

            cv2.waitKey(1) & 0xFF == ord('q')

        cap.release()
        cv2.destroyAllWindows()
