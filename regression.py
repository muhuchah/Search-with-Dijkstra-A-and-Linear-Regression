import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Regression:
    n = m = p = nabla = 0
    X = y = theta = []
    
    def __init__(self, input_dataset, output_dataset, number_of_iterations, learning_rate):
        self.X = input_dataset
        self.y = output_dataset
        self.m = len(self.X)
        # add fake feature
        for i in range(self.m):
            self.X[i].append(1)
        self.n = len(self.X[0])
        self.p = number_of_iterations
        self.nabla = learning_rate
        self.theta = [0]*self.n
        self.gradiant_decent()

    def h(self, idx):
        res = 0
        for i in range(self.n):
            res += self.X[idx][i]*self.theta[i]
        return res

    def J(self):
        res = 0
        for i in range(self.m):
            res += (self.h(self.X[i])-self.y[i])**2
        res /= 2*self.m
        return res

    def gradiant(self):
        gradiant_vector = [0]*self.n
        for j in range(self.n):
            for i in range(self.m):
                gradiant_vector[j] += (self.h(i)-self.y[i])*self.X[i][j]
            gradiant_vector[j] /= self.m
        return gradiant_vector
        
    def gradiant_decent(self):
        for i in range(self.p):
            for j in range(self.n):
                self.theta[j] -= self.nabla*self.gradiant()[j]
    
    def predict(self, x):
        if len(x) < self.n:
            x.append(1)
        res = 0
        for i in range(self.n):
            res += self.theta[i]*x[i]
        return res
            
        
    
    



# number_of_iterations = 1000
# learning_rate = 0.1
# input_dataset = [
#     [1],
#     [2],
#     [3],
#     [4],
#     [5],
#     [6]
# ]
# output_dataset = [1, 2, 3, 4, 5, 6]
# reg = Regression(input_dataset, output_dataset, number_of_iterations, learning_rate)

# test = []
# for i in range(6, 10):
#     test.append(reg.h([i]))

# print(test)
df = pd.read_csv("/home/pouya/downloads/Flight_Price_Dataset_Q2.csv")
# lb = LabelEncoder()
# df["Mapping"] = lb.fit_transform(df["class"])

# df1 = pd.concat([df, departure_time_mapping], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
departure_time_mapping = {
    "Early_Morning": 0,
    "Morning": 1,
    "Afternoon": 2,
    "Night": 3, 
    "Late_Night": 4
}
stops_mapping = {
    "zero": 0,
    "one": 1,
    "two_or_more": 2
}
class_mapping = {
    "Economy": 0,
    "Business": 1
}
df["departure_time"] = df["departure_time"].map(departure_time_mapping)
df["stops"] = df["stops"].map(stops_mapping)
df["arrival_time"] = df["arrival_time"].map(departure_time_mapping)
df["class"] = df["class"].map(class_mapping)
df = df.dropna()
df = df.reset_index(drop=True)
y = df["price"]
X = df.drop("price", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

learning_rate = 0.1
number_of_iterations = 10
reg = Regression(X_train.values.tolist(), y_train.values.tolist(), number_of_iterations, learning_rate)


def test(X_test, y_test):
    y_hat = [0]*len(X_test)
    for i in range(len(X_test)):
        y_hat[i] = reg.predict(X_test[i])
        
    print(y_hat)
    print(mean_absolute_error(y_hat, y_test))
    print(mean_squared_error(y_hat, y_test))
    print(r2_score(y_hat, y_test))
    


test(X_test.values.tolist(), y_test.values.tolist())