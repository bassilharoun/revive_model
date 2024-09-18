from shared.data_processing import Data_Loader
from shared.graph import Graph


class TrainExercise:
    def __init__(self, exercise_path, epoch, learning_rate, batch_size, random_seed=42):
        self.exercise = exercise_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.exercise = exercise_path
        self.epoch = epoch
        self.random_seed = random_seed

    def load_data(self, exercise_path):
        data_loader = Data_Loader(exercise_path)
        graph = Graph(len(data_loader.body_part))
        return data_loader, graph

    def split_data(self, data_loader):
        train_x, valid_x, train_y, valid_y = train_test_split(
            data_loader.scaled_x,
            data_loader.scaled_y,
            test_size=0.2,
            random_state=self.random_seed,
        )
        print("Training instances: ", len(train_x))
        print("Validation instances: ", len(valid_x))

        return train_x, valid_x, train_y, valid_y

    def train(self, train_x, valid_x, train_y, valid_y, graph):
        # algorithm = Sgcn_Lstm(train_x, train_y, graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2, lr = args.lr, epoach=args.epoch, batch_size=args.batch_size)
        if (self.train_x is None) or (self.train_y is None):
            print("Data not loaded")
            return

        self.algorithm = Sgcn_Lstm(
            train_x,
            train_y,
            valid_x,
            valid_y,
            graph.AD,
            graph.AD2,
            lr=learning_rate,
            epoach=self.epoch,
            batch_size=batch_size,
        )
        self.history = self.algorithm.train()
        return self

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def Evaluate(self, model, valid_x, valid_y, data_loader):
        y_pred = model.prediction(valid_x)
        y_pred = data_loader.sc2.inverse_transform(y_pred)
        valid_y = data_loader.sc2.inverse_transform(valid_y)

        test_dev = abs(valid_y - y_pred)
        mean_abs_dev = np.mean(test_dev)

        mae = mean_absolute_error(valid_y, y_pred)
        rms_dev = sqrt(mean_squared_error(y_pred, valid_y))
        mse = mean_squared_error(valid_y, y_pred)
        mape = self.mean_absolute_percentage_error(valid_y, y_pred)

        return mae, mse, mape, rms_dev, mean_abs_dev

    def print_results(self, mae, mse, mape, rms_dev, mean_abs_dev):
        print("Mean absolute deviation:", mae)
        print("RMS deviation:", rms_dev)
        print("MSE:", mse)
        print("MAPE: ", mape)

    def Run(self):
        data_loader, self.graph = self.load_data(self.exercise)
        train_x, self.valid_x, self.train_y, self.valid_y = self.split_data(data_loader)
        model = self.train(
            train_x, self.valid_x, self.train_y, self.valid_y, self.graph
        )
        evaluate = self.Evaluate(model, self.valid_x, self.valid_y, data_loader)
        self.print_results(*evaluate)