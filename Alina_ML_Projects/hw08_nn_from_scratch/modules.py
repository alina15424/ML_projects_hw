import numpy as np
import scipy as sp
import scipy.signal
import skimage
import skimage.util

class Module(object):
    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 
        
        output = module.forward(input)
    
    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        gradInput = module.backward(input, gradOutput)
    """
    def __init__ (self):
        self.output = None
        self.gradInput = None
        self.training = True
    
    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self,input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.
        
        This includes 
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput
    

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.
        
        Make sure to both store the data in `output` field and return it. 
        """
        
        # The easiest case:
            
        # self.output = input 
        # return self.output
        
        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input. 
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.
        
        The shape of `gradInput` is always the same as the shape of `input`.
        
        Make sure to both store the gradients in `gradInput` field and return it.
        """
        
        # The easiest case:
        
        # self.gradInput = gradOutput 
        # return self.gradInput
        
        pass   
    
    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass
    
    def zeroGradParameters(self): 
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass
        
    def getParameters(self):
        """
        Returns a list with its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
        
    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
    
    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True
    
    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Module"


class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially. 
         
         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`. 
    """
    
    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []
   
    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:
        
            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})   
            
            
        Just write a little loop. 
        """

        # Your code goes here. ################################################
        y = input
        for module in self.modules:
            y = module.forward(y)
        self.output = y
        return self.output


    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:
            
            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)   
            gradInput = module[0].backward(input, g_1)   
             
             
        !!!
                
        To each module you need to provide the input, module saw while forward pass, 
        it is used while computing gradients. 
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass) 
        and NOT `input` to this Sequential module. 
        
        !!!
        
        """
        # Your code goes here. ################################################
        n = len(self.modules)
        g = gradOutput
        backward_modules  = self.modules[1:][::-1] #without module[0], it will be used separately for function output
        for i, module in enumerate(backward_modules):
            g = module.backward(self.modules[n - 2 - i].output, g)
        self.gradInput = self.modules[0].backward(input, g)
        return self.gradInput
      

    def zeroGradParameters(self): 
        for module in self.modules:
            module.zeroGradParameters()
    
    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]
    
    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string
    
    def __getitem__(self,x):
        return self.modules.__getitem__(x)
    
    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()
    
    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


class Linear(Module):
    """
    A module which applies a linear transformation 
    A common name is fully-connected layer, InnerProductLayer in caffe. 
    
    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
       
        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        # Your code goes here. ################################################
        # self.output = ...
        self.output = np.add(np.dot(input,np.transpose(self.W)), self.b)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        # self.gradInput = ...
        self.gradInput = np.dot(gradOutput, self.W)
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        # Your code goes here. ################################################
        # self.gradW = ... ; self.gradb = ...
        self.gradW = np.dot(np.transpose(gradOutput),input)
        self.gradb = np.dot(np.ones(input.shape[0]),gradOutput)
        
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q

class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        
        # Your code goes here. ################################################
        np.exp(self.output,  out = self.output)
        np.divide(self.output,np.sum(self.output, axis=1).reshape(-1,1), out = self.output)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        gradOutput = np.multiply(gradOutput,self.output)
        self.gradInput = np.subtract(gradOutput,np.multiply(self.output,np.sum(gradOutput, axis=1, keepdims=True).reshape(-1,1)))
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"


class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        
        # Your code goes here. ################################################
        np.subtract(self.output, np.log(np.sum(np.exp(self.output), axis=1).reshape(-1,1)), out = self.output)        
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.zeros(input.shape)
        for i in range(input.shape[0]):
            np.dot(gradOutput[i], np.subtract(np.eye(input.shape[1]),np.exp(self.output)[i]), out = self.gradInput[i])
        return self.gradInput
    
    def __repr__(self):
        return "LogSoftMax"


class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = 0 
        self.moving_variance = 0
        
    def updateOutput(self, input):
        # Your code goes here. ################################################
        # use self.EPS please
        self.n = input.shape[0]
        self.output = np.zeros_like(input)
        if self.training == True:
            self.batch_mean = np.mean(input, axis=0)
            self.batch_variance = np.var(input, axis=0)
            self.moving_mean = self.moving_mean * self.alpha + self.batch_mean * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + self.batch_variance * (1 - self.alpha)
            np.divide(np.subtract(input,self.batch_mean),np.sqrt(np.add(self.batch_variance,self.EPS)), out = self.output)
        else:
            np.divide(np.subtract(input,self.moving_mean),np.sqrt(np.add(self.moving_variance,self.EPS)), out = self.output)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.zeros_like(input)
        normalized = np.zeros_like(input)
        np.divide(np.subtract(input,self.batch_mean),(np.sqrt(np.add(self.batch_variance,self.EPS))), out = normalized)
        np.multiply(np.divide(1,np.multiply(np.sqrt(np.add(self.batch_variance,self.EPS)),self.n)),np.subtract(np.subtract(np.multiply(self.n,gradOutput),gradOutput.sum(axis=0)),np.multiply(normalized,np.sum(np.multiply(gradOutput,normalized), axis=0))), out = self.gradInput)
        return self.gradInput
    
    def __repr__(self):
        return "BatchNormalization"



class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output
        
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput*input, axis=0)
    
    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def getParameters(self):
        return [self.gamma, self.beta]
    
    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = None
        
    def updateOutput(self, input):
        # Your code goes here. ################################################
        self.output = np.zeros(input.shape)
        self.mask = np.random.binomial(1, 1. - self.p, input.shape)
        if self.training == True:
            np.multiply(input, self.mask, out = self.output)
            np.divide(self.output,1. - self.p, out = self.output)
        else:
            self.output = input
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.zeros(input.shape)
        if self.training == True:
            np.multiply(gradOutput, self.mask, out = self.gradInput)
            np.divide(self.gradInput,(1. - self.p), out = self.gradInput)
        else:
            self.gradInput = gradOutput
        return self.gradInput
        
    def __repr__(self):
        return "Dropout"


class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"


class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, input):
        # Your code goes here. ################################################
        self.output = input.copy()
        self.output[self.output < 0] = self.output[self.output < 0]*self.slope
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        boolpos = input >= 0
        pos = gradOutput*(boolpos)
        boolneg = input < 0
        neg = gradOutput*boolneg*self.slope
        self.gradInput = pos + neg
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"
    
    


class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        
        self.alpha = alpha
        
    def updateOutput(self, input):
        # Your code goes here. ################################################
        self.output = input.copy()
        self.output[self.output < 0] = (np.exp(self.output[self.output < 0])-1)*self.alpha
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        boolpos = input >= 0
        pos = gradOutput*(boolpos)
        boolneg = input < 0
        neg = gradOutput*np.exp(input)*boolneg*self.alpha
        self.gradInput = pos + neg
        return self.gradInput
    
    def __repr__(self):
        return "ELU"



class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, input):
        # Your code goes here. ################################################
        self.output = np.log(np.exp(input)+1)
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.divide(gradOutput, (np.exp(-input)+1))
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"


class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None
        
    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function 
            associated to the criterion and return the result.
            
            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result. 

            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)
    
    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput   

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Criterion"


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()
        
    def updateOutput(self, input, target):   
        self.output = np.sum(np.power(input - target,2)) / input.shape[0]
        return self.output 
 
    def updateGradInput(self, input, target):
        self.gradInput  = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"



class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15
    def __init__(self):
        a = super(ClassNLLCriterionUnstable, self)
        super(ClassNLLCriterionUnstable, self).__init__()
        
    def updateOutput(self, input, target): 
        
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        
        # Your code goes here. ################################################
        self.N = input.shape[0]
        self.output = np.negative(np.divide(np.sum(np.multiply(target, np.log(input_clamp))),self.N))
        return self.output

    def updateGradInput(self, input, target):
        
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
                
        # Your code goes here. ################################################
        self.gradInput = np.negative(np.divide(np.divide(target, input_clamp),self.N))
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterionUnstable"



class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()
        
    def updateOutput(self, input, target): 
        # Your code goes here. ################################################
        self.N = input.shape[0]
        self.output = np.negative(np.divide(np.sum(np.multiply(target, input)),self.N))
        return self.output

    def updateGradInput(self, input, target):
        # Your code goes here. ################################################
        self.gradInput = np.negative(np.divide(target,self.N))
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterion"



class ClassContrastiveCriterion(Criterion):
    def __init__(self, M):
        a = super(ClassContrastiveCriterion, self)
        super(ClassContrastiveCriterion, self).__init__()
        self.M = M #margin for punishing the negative pairs
        
    def updateOutput(self, input, target): 
        # Your code goes here. ################################################
            
        return self.output

  
    def updateGradInput(self, input, target):
        # Your code goes here. ################################################
        return self.gradInput
    
    def __repr__(self):
        return "ClassContrastiveCriterion"



# Optimizers
def sgd_momentum(variables, gradients, config, state):
    state.setdefault('accumulated_grads', {})
    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))
            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)
            current_var -= old_grad
            var_index += 1

def adam_optimizer(variables, gradients, config, state):
    state.setdefault('m', {})
    state.setdefault('v', {})
    state.setdefault('t', 0)
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()

    var_index = 0
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2'] ** state['t']) / (1 - config['beta1'] ** state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            m = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            v = state['v'].setdefault(var_index, np.zeros_like(current_grad))

            m *= config['beta1']
            m += (1 - config['beta1']) * current_grad

            v *= config['beta2']
            v += (1 - config['beta2']) * (current_grad ** 2)

            current_var -= lr_t * m / (np.sqrt(v) + config['epsilon'])
            var_index += 1

# Conv2d layer
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size  # Размер ядра должен быть нечетным

        stdv = 1. / np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        """
        Forward pass: computes convolution with the given kernel and adds bias.
        """
        pad_size = self.kernel_size // 2
        padded_input = np.pad(input, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='constant')
        self.output = np.zeros((input.shape[0], self.out_channels, input.shape[2], input.shape[3]))

        for b in range(input.shape[0]):  # Loop over the batch dimension
            for i in range(self.out_channels):
                for j in range(self.in_channels):
                    self.output[b, i] += sp.signal.correlate(padded_input[b, j], self.W[i, j], mode='valid')
                self.output[b, i] += self.b[i]  # Add bias

        return self.output

    def updateGradInput(self, input, gradOutput):
        """
        Backward pass: computes gradient with respect to the input.
        """
        pad_size = self.kernel_size // 2
        padded_gradOutput = np.pad(gradOutput, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='constant')
        self.gradInput = np.zeros_like(input)

        for b in range(input.shape[0]):  # Loop over the batch dimension
            for i in range(self.in_channels):
                for j in range(self.out_channels):
                    flipped_kernel = self.W[j, i, ::-1, ::-1]
                    self.gradInput[b, i] += sp.signal.correlate(padded_gradOutput[b, j], flipped_kernel, mode='valid')

        return self.gradInput


    def accGradParameters(self, input, gradOutput):
        """
        Накопление градиентов по параметрам (весу и смещению).
        """
        pad_size = self.kernel_size // 2
        padded_input = np.pad(input, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='constant')

        for i in range(self.out_channels):
            for j in range(self.in_channels):
                self.gradW[i, j] += np.sum([
                    sp.signal.correlate(padded_input[b, j], gradOutput[b, i], mode='valid')
                    for b in range(input.shape[0])
                ], axis=0)

        self.gradb += np.sum(gradOutput, axis=(0, 2, 3))

    def zeroGradParameters(self):
        """
        Обнуляет градиенты для параметров.
        """
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        """
        Возвращает параметры слоя (веса и смещения).
        """
        return [self.W, self.b]

    def getGradParameters(self):
        """
        Возвращает градиенты параметров слоя.
        """
        return [self.gradW, self.gradb]

    def __repr__(self):
        """
        Читаемое представление слоя.
        """
        s = self.W.shape
        q = f'Conv2d {s[1]} -> {s[0]}'
        return q

# MaxPool2d layer
class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.gradInput = None

    def updateOutput(self, input):
        N, C, H, W = input.shape
        k = self.kernel_size
        assert H % k == 0 and W % k == 0
        H_new = H // k
        W_new = W // k
        input_reshaped = input.reshape(N, C, H_new, k, W_new, k)
        input_reshaped = input_reshaped.transpose(0, 1, 2, 4, 3, 5)
        input_reshaped = input_reshaped.reshape(N, C, H_new, W_new, k * k)
        self.output = np.max(input_reshaped, axis=-1)
        self.max_indices = np.argmax(input_reshaped, axis=-1)
        return self.output

    def updateGradInput(self, input, gradOutput):
        N, C, H, W = input.shape
        k = self.kernel_size
        H_new = H // k
        W_new = W // k
        gradInput = np.zeros_like(input)
        gradInput_reshaped = gradInput.reshape(N, C, H_new, k, W_new, k)
        gradInput_reshaped = gradInput_reshaped.transpose(0, 1, 2, 4, 3, 5)
        gradInput_flat = gradInput_reshaped.reshape(-1, k * k)
        gradOutput_flat = gradOutput.flatten()
        indices = np.arange(gradInput_flat.shape[0])
        max_indices_flat = self.max_indices.flatten()
        gradInput_flat[indices, max_indices_flat] = gradOutput_flat
        gradInput_reshaped = gradInput_flat.reshape(N, C, H_new, W_new, k, k)
        gradInput_reshaped = gradInput_reshaped.transpose(0, 1, 2, 4, 3, 5)
        self.gradInput = gradInput_reshaped.reshape(N, C, H, W)
        return self.gradInput

    def __repr__(self):
        return 'MaxPool2d, kern %d, stride %d' % (self.kernel_size, self.kernel_size)

# Flatten layer
class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def updateOutput(self, input):
        self.output = input.reshape(len(input), -1)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput.reshape(input.shape)
        return self.gradInput

    def __repr__(self):
        return "Flatten"