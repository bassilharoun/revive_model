from realtime import RealTimeApp
import os

from ui import UI


exercise_folder = "data/dataset/KIMORE/Kimore_ex5"
model_path = "/home/modsyan/work/automatic-evaluation-of-physical-therapy-exercises/trained/stgcn-lstm/ex5/my_model_trained_exercisetest500.keras"

if (
    not os.path.exists(exercise_folder)
    or not os.path.exists(model_path)
):
    print("Please provide valid paths for exercise_folder, model_path and video_path")
    exit()

# None if video from webcam
video_path = "data/test/VID20240328063258.mp4"


ui = UI()
app = RealTimeApp(exercise_folder, model_path, ui, video_path)
app.run(batch_size=200)
