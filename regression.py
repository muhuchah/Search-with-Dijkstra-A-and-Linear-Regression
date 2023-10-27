
class regression:
    n = m = p = nabla = 0
    X = y = theta = []
    
    def __init__(self, dimention, number_of_samples, number_of_iterations, learning_rate):
        self.m = number_of_samples
        self.n = dimention
        self.p = number_of_iterations
        self.nabla = learning_rate
        self.X = self.y = self.theta = []

    def h(self, theta, X):
        return self.theta.dot(self.X)

    def J(self, theta):
        res = 0
        for i in range(self.m):
            res += (self.h(theta, self.X[i])-self.y[i])**2
        res /= 2*self.m
        return res
    def gradiant(self, theta):
        gradiant_vector = [0]*self.n
        for j in range(self.n):
            for i in range(self.m):
                gradiant_vector[j] += (self.h(theta, self.X[i])-self.y[i])*self.X[i][j]
            gradiant_vector[j] /= self.m
        return gradiant_vector
        
    def gradiant_decent(self):
        for i in range(self.p):
            theta -= self.nabla*self.gradiant(self.theta)
    