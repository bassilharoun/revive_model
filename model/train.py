from model.exercise_train import TrainExercise

exercise = "data/KIMORE/Kimore_ex5"
learning_rate = 0.0001
epoch = 500
batch_size = 10
trainer = TrainExercise(exercise, epoch, learning_rate, batch_size)
trainer.Run()