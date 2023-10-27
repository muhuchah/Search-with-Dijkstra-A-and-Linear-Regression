import pandas as pd
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

    def h(self, x):
        res = 0
        # add fake feature
        x.append(1)
        for i in range(self.n):
            res += x[i]*self.theta[i]
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
                gradiant_vector[j] += (self.h(self.X[i])-self.y[i])*self.X[i][j]
            gradiant_vector[j] /= self.m
        return gradiant_vector
        
    def gradiant_decent(self):
        for i in range(self.p):
            for j in range(self.n):
                self.theta[j] -= self.nabla*self.gradiant()[j]
    
    

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
# output_dataset = [1, 1, 1, 1, 1, 1]
# reg = Regression(input_dataset, output_dataset, number_of_iterations, learning_rate)

# test = []
# for i in range(6, 10):
#     test.append(reg.h([i]))

# print(test)
df = pd.read_csv("/home/pouya/downloads/Flight_Price_Dataset_Q2.csv")
df