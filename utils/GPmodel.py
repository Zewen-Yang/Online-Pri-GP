# Copyright (c) by Zewen Yang under GNU General Public License v3 (GPLv3)
# Last modified: Zewen Yang 11/2024

import numpy as np

class GPmodel:
    def __init__(self, indivDataThersh, x_dim, y_dim, 
                 sigmaN, sigmaF, sigmaL, 
                 priorFunc, agentIndex=None):
        self.agentIndex = agentIndex
        self.DataQuantity = 0
        self.indivDataThersh = indivDataThersh  # data limit per local GP
        self.x_dim = x_dim  # dimensionality of training input x
        self.y_dim = y_dim  # dimensionality of training output y
        # hyperparameters
        self.sigmaN = sigmaN 
        self.sigmaF = sigmaF
        self.sigmaL = sigmaL

        # data storage
        self.X = np.zeros([indivDataThersh, x_dim], dtype=float)
        self.Y = np.zeros([indivDataThersh, y_dim], dtype=float)

        # kernel matrix
        self.K = np.zeros([y_dim, indivDataThersh, indivDataThersh], dtype=float)
        self.K_border = np.zeros([indivDataThersh, y_dim], dtype=float)
        self.K_corner = np.zeros([1, y_dim], dtype=float)
        self.KinvKborder = np.zeros([indivDataThersh, y_dim], dtype=float)

        self.priorFunc = priorFunc
        self.priorError_list = np.empty((0, y_dim), dtype=float)

        self.predictTimes = 0
        

    

    def kernel(self, Xi: np.ndarray, Xj: np.ndarray, dim: int) -> np.ndarray:
        """
        ARD Squared Exponential Kernel
        
        Args:
            X1: Array of shape (N, D)
            X2: Array of shape (M, D)
            sigmaF: Signal variance (output scale), shape (y_dim, 1)
            sigmaL: Length scales for each dimension, shape (x_dim, y_dim)
        
        Returns:
            Kernel matrix of shape (N, M) or (N, N) if X1 == X2
        """
        # Calculate pairwise distances
        diff = Xi[:, None, :] - Xj[None, :, :]  # Shape: (N, M, D)
    
        # Scale differences by length scales
        scaled_diff = diff / self.sigmaL[:, dim]  # Using first column of sigmaL for single output
        
        # Calculate squared distances and sum over features
        squared_dist = np.sum(scaled_diff ** 2, axis=2)
        
        # Apply the exponential
        kernel_matrix = (self.sigmaF[0, dim] ** 2) * np.exp(-0.5 * squared_dist)
        
        return kernel_matrix


    def addDataEntire(self, X_in, Y_in):
        # Check sample size match
        if X_in.shape[0] != Y_in.shape[0]:
            raise ValueError(f"Input shapes don't match: X has {X_in.shape[0]} samples while Y has {Y_in.shape[0]} samples")
        
        # Check input dimensions match model dimensions
        if X_in.shape[1] != self.x_dim:
            raise ValueError(f"X input dimension mismatch: expected {self.x_dim}, got {X_in.shape[1]}")
        if Y_in.shape[1] != self.y_dim:
            raise ValueError(f"Y input dimension mismatch: expected {self.y_dim}, got {Y_in.shape[1]}")
        
        # Process data
        minDataQuantity = min(X_in.shape[0], self.indivDataThersh)
        self.DataQuantity = minDataQuantity
        self.X[range(minDataQuantity),:] = X_in[range(minDataQuantity),:]
        self.Y[range(minDataQuantity),:] = Y_in[range(minDataQuantity),:]



    def updateKmatEntire(self):
        X_set = self.X[range(self.DataQuantity), :]
        for dim in range(self.y_dim):
            K = self.kernel(X_set, X_set, dim)
            K_noise =  K + self.sigmaN[0, dim] ** 2 * np.eye(self.DataQuantity)
            self.K[dim, 
                   0 : self.DataQuantity, 
                   0 : self.DataQuantity] = K_noise


    def addDataOnce(self, x, y):
        # x, y are 2D arrays with one row
        self.X[self.DataQuantity, :] = x[0]
        self.Y[self.DataQuantity, :] = y[0]
        self.DataQuantity = self.DataQuantity + 1



    def updateKmatOnce(self):
        # Get the new data point 
        x_new = self.X[self.DataQuantity-1:self.DataQuantity, :]
        # Get all previous data points
        X_old = self.X[0:self.DataQuantity-1, :]
        
        for dim in range(self.y_dim):
            temp_K_border = self.kernel(X_old, x_new, dim).flatten()
            temp_K_corner = self.kernel(x_new, x_new, dim) + self.sigmaN[0, dim] ** 2
            self.K_border[0:self.DataQuantity-1, dim] = temp_K_border
            self.K_corner[0, dim] = temp_K_corner
            # Set border values without reshaping
            self.K[dim, :self.DataQuantity-1, self.DataQuantity-1] = temp_K_border
            self.K[dim, self.DataQuantity-1, :self.DataQuantity-1] = temp_K_border
            # Set corner value
            self.K[dim, self.DataQuantity-1, self.DataQuantity-1] = temp_K_corner
                

    def errorRecord(self,x,y):
            temp_error = np.absolute(self.priorFunc(x) - y)
            self.priorError_list = np.vstack((self.priorError_list, temp_error))


    def predict_initial(self, x_test):
            self.predictTimes = self.predictTimes + 1
            """Prediction method when no data is available"""
            if len(x_test.shape) == 1:
                x_test = x_test.reshape(1, -1)
            
            mean = np.zeros((1, self.y_dim))
            var = np.zeros((1, self.y_dim))
            
            mean = self.priorFunc(x_test)
            for dim in range(self.y_dim):
                var[:, dim] = self.sigmaN[0, dim] ** 2
            return mean, var
    
    def predict(self, x_test):
        self.predictTimes = self.predictTimes + 1
        """Regular GP prediction method when data is available"""
        # Ensure x_test is 2D
        # if len(x_test.shape) == 1:
        #     x_test = x_test.reshape(1, -1)
        
        mean = np.zeros((1, self.y_dim))
        var = np.zeros((1, self.y_dim))

        # Get training data
        X_train = self.X[:self.DataQuantity]
        Y_train = self.Y[:self.DataQuantity]
        
        for dim in range(self.y_dim):
            # Compute kernel between test and training points
            temp_K_border = self.kernel(X_train, x_test, dim).flatten()  # (N_train,)
            
            # Compute kernel between test points
            temp_K_starstar = self.kernel(x_test, x_test, dim) 
            
            # Get the inverse of K using numpy's solve
            K_inv_y = np.linalg.solve(
                self.K[dim, :self.DataQuantity, :self.DataQuantity],
                (Y_train[:, dim] - self.priorFunc(X_train)[:, dim])
            )
            
            # Compute predictive mean
            mean[:, dim] = temp_K_border @ K_inv_y + self.priorFunc(x_test)[:, dim]
            
            
            # Compute predictive variance
            K_inv_k_star = np.linalg.solve(
                self.K[dim, :self.DataQuantity, :self.DataQuantity],
                temp_K_border.T
            )
            var[:, dim] = temp_K_starstar - temp_K_border @ K_inv_k_star
        
        return mean, var
