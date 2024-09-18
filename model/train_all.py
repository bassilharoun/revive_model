from model.exercise_train import TrainExercise


exercises_list = [
    ["data/KIMORE/Kimore_ex1", 0.0001, 500, 10],
    ["data/KIMORE/Kimore_ex2", 0.0001, 500, 10],
    ["data/KIMORE/Kimore_ex3", 0.0001, 500, 10],
    ["data/KIMORE/Kimore_ex4", 0.0001, 500, 10],
    ["data/KIMORE/Kimore_ex5", 0.0001, 500, 10],
]

for exercise, learning_rate, epoch, batch_size in exercises_list:
    trainer = TrainExercise(exercise, epoch, learning_rate, batch_size)
    trainer.Run()