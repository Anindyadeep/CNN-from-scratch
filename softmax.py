import numpy as np

class Softmax:
    def __init__ (self, image_len, nodes):

        self.weights = np.random.randn(image_len, nodes)/image_len
        self.bias = np.zeros(shape=(nodes))

    def softmax_forward(self, image):
        # storing the input shape of the maxpool image
        # NOTE: THIS IS VERY IMP AS WE ARE STORING THIS COZ, AFTER CALCULATING dL/dX WE HAVE TO
        #       RESHAPE IT TO THE PREVIOUS MATRIX OF THE MAXPOOL-ed IMAGE SIZE FOR THE BACKPROP IN MAXPOOL AND THE CONV LAYER
        
        self.last_input_shape = image.shape

        image = image.flatten()
        # storing/caching the last input (flatten)
        self.last_input = image

        input_len, nodes = self.weights.shape
        hidden_unit = np.dot(image, self.weights)+self.bias
        # storing these intermediate value Z
        self.last_hidden_unit = hidden_unit

        soft_all = np.exp(hidden_unit)
        return soft_all/np.sum(soft_all)

    def soft_backward(self, dL_dOut, learning_rate):
        '''
        This is the backward for only the non-zero value of the gradients of the probabilitis i.e. for i = c
        so the gradients will look like --> [0,0,0,0, ... , -1/p[c], 0, 0, ..., 0]
        The further computation is done for that index. 
        So the things are doing as follows:

        1. we first compute the dOut/dt = -{(last_probs[i]) * last_probs}/ S**2
        (WHERE last_probs IS THE PROB OF EACH CLASS)
        so if the probabilities are (Out) = [0.99,1.00,0.998,1.0014,0.999,1.00,0.999,1.00,0.998,0.99]

        2. the i = c (class) here is 5 (let)
        so we will take the grads[5]

        where S = sum(Out) = 9.998243 (let)

        3. dOut/dt = -{Out[5] * Out}/S**2  [i != k]
        = -([1.00,1.00,...., 1.00] * [0.99,1.00,0.998,1.0014,0.999,1.00,0.999,1.00,0.998,0.99])/ S**2
        =   [-0.010,-0.016,-0.009,-0.010,-0.01,-0.010,-0.010,-0.0100, -0.0099,-0.010]

        4. Now when i == k for that time 
        dOut_dt[i == k] = last_probs[i] * (S-last_probs[i])/(S**2)

        dOut_dt[5] = Out[5] * (S-Out[5])/(S**2)
                   = 1.00120 * (9.998243)
        
        Now the value of dOut_dt[5] will be replaced with this new value computed just before
        Hence it becomes: 
        [-0.010,-0.016,-0.009,-0.010,(1.00120 * (9.998243)),-0.010,-0.010,-0.0100, -0.0099,-0.010
        '''

        ''' 
        This process will be repeated for each of the images iteratively comming from the softmax layer
        '''
        for i, grad in enumerate(dL_dOut):
            if grad == 0:
                continue
            # getting the Z values and converting them to probabilities
            last_probs = np.exp(self.last_hidden_unit)
            #print("THE VALUE OF THE PROBABILITIES FOR EACH CLASS", last_probs, "\n")
            # summing the probs
            S = np.sum(last_probs)
            #print("FOR i = ", i)
            #print("THE VALUE OF SUM: ", S, "\n")

            # gradients of out[i] (non zero) against total
            dOut_dt = -last_probs[i]*last_probs/(S**2)
            #print("VALUE OF dOut_dt ", dOut_dt, "\n")
            dOut_dt[i] = last_probs[i] * (S-last_probs[i])/(S**2)

            #print("VALUE OF dOut_dt(i=k) ", dOut_dt[i], "\n")
            #print("FINAL VALUE OF dOut_dt bacomes: ", dOut_dt, "\n")\

            # t = Z (here)
            dt_dW = self.last_input
            dt_db = 1
            dt_dinputs = self.weights

            dL_dt = grad*dOut_dt
            dL_dW = (dL_dt[np.newaxis].T @ dt_dW[np.newaxis]).T 
            dL_db = dL_dt * dt_db
            dL_dinputs = dt_dinputs @ dL_dt

            #print("dL_dZ shape: ", dL_dt.shape)
            #print("dL_dW shape: ", dL_dW.shape)
            #print("dL_db shape: ", dL_db.shape)
            #print("dL_dinputs shape: ", dL_dinputs.shape)

            # updating the weights and bias of the softmax layer
            self.weights -= learning_rate * dL_dW
            self.bias  -= learning_rate * dL_db

            # finally return the dL/dX which will be used in the previous outputs
            return dL_dinputs.reshape(self.last_input_shape)