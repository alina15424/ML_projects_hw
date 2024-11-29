import numpy as np
import scipy as sp

"""**Module** is an abstract class which defines fundamental methods necessary for a training a neural network. You do not need to change anything here, just read the comments."""

class Module(object):
    """
    Base class for all neural network modules. This is a template for implementing
    specific layers like Linear, ReLU, SoftMax, etc.
    """
    def __init__(self):
        self.output = None  # Stores the output of the forward pass
        self.gradInput = None  # Stores the gradient with respect to the input
        self.training = True  # Indicates if the module is in training mode

    def forward(self, input):
        """
        Computes the output of the module for a given input.
        """
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        """
        Computes the gradient of the loss with respect to the module's input and parameters.
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, input, target):
        """
        Вычисляет Negative Log-Likelihood (NLL):
        output = - log(input[target])
        """
        # Преобразуем target в целые числа, если это не так
        target = target.astype(int)

        # Предотвращение логарифма от 0
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        # Вычисляем NLL
        self.output = -np.mean(np.log(input_clamp[np.arange(target.shape[0]), target]))
        return self.output



    def updateGradInput(self, input, gradOutput):
        """
        Computes and returns the gradient of the module with respect to its input.
        """
        self.gradInput = gradOutput  # Example case: identity gradient
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        """
        Updates the gradient of the module's parameters with respect to the loss.
        Override this method for modules with learnable parameters.
        """
        pass

    def zeroGradParameters(self):
        """
        Resets the gradients of the module's parameters to zero.
        Override this method for modules with learnable parameters.
        """
        pass

    def getParameters(self):
        """
        Returns a list of the module's learnable parameters.
        """
        return []

    def getGradParameters(self):
        """
        Returns a list of the gradients of the module's parameters.
        """
        return []

    def train(self):
        """
        Sets the module in training mode.
        """
        self.training = True

    def evaluate(self):
        """
        Sets the module in evaluation mode.
        """
        self.training = False

    def __repr__(self):
        """
        Returns a string representation of the module.
        """
        return "Module"

"""# Sequential container

**Define** a forward and backward pass procedures.
"""

class Sequential(Module):
    """
    This class implements a container, which processes `input` data sequentially.

    `input` is processed by each module (layer) in self.modules consecutively.
    The resulting array is called `output`.
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []
        self.outputs = []  # To store outputs of each module during forward pass

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
        self.outputs = []  # Clear previous outputs
        current_output = input
        for module in self.modules:
            current_output = module.forward(current_output)
            self.outputs.append(current_output)  # Store output for backward pass
        self.output = current_output
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

        To each module, you need to provide the input that the module saw during the forward pass,
        as it is used while computing gradients.
        Make sure that the input for the i-th layer is the output of module i-1 (just the same input as in forward pass)
        and NOT the input to this Sequential module.

        !!!
        """
        current_grad = gradOutput
        for idx in reversed(range(len(self.modules))):
            module = self.modules[idx]
            if idx == 0:
                module_input = input
            else:
                module_input = self.outputs[idx - 1]
            current_grad = module.backward(module_input, current_grad)
        self.gradInput = current_grad
        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        params = []
        for module in self.modules:
            module_params = module.getParameters()
            if module_params:
                params.extend(module_params)
        return params

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        grad_params = []
        for module in self.modules:
            module_grads = module.getGradParameters()
            if module_grads:
                grad_params.extend(module_grads)
        return grad_params

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self, x):
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






"""# Layers

## 1. Linear transform layer
Also known as dense layer, fully-connected layer, FC-layer, InnerProductLayer (in caffe), affine transform
- input:   **`batch_size x n_feats1`**
- output: **`batch_size x n_feats2`**
"""

class Linear(Module):
    """
    A module which applies a linear transformation.
    A common name is fully-connected layer, InnerProductLayer in Caffe.

    The module works with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        # This is a nice initialization
        stdv = 1. / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        """
        Computes the output of the linear layer.
        output = input * W^T + b
        """
        self.output = input.dot(self.W.T) + self.b  # (n_samples, n_out)
        return self.output

    def updateGradInput(self, input, gradOutput):
        """
        Computes the gradient of the loss with respect to the input of the linear layer.
        gradInput = gradOutput * W
        """
        self.gradInput = gradOutput.dot(self.W)  # (n_samples, n_in)
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        """
        Accumulates the gradients of the weights and biases.
        gradW = gradOutput^T * input
        gradb = sum of gradOutput along samples
        """
        self.gradW += gradOutput.T.dot(input)  # (n_out, n_in)
        self.gradb += gradOutput.sum(axis=0)  # (n_out,)

    def zeroGradParameters(self):
        """
        Resets the gradients of the weights and biases to zero.
        """
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        """
        Returns the weights and biases as a list.
        """
        return [self.W, self.b]

    def getGradParameters(self):
        """
        Returns the gradients of the weights and biases as a list.
        """
        return [self.gradW, self.gradb]

    def __repr__(self):
        """
        Returns a string representation of the module.
        """
        s = self.W.shape
        q = 'Linear %d -> %d' % (s[1], s[0])
        return q

"""## 2. SoftMax
- input:   **`batch_size x n_feats`**
- output: **`batch_size x n_feats`**

$\text{softmax}(x)_i = \frac{\exp x_i} {\sum_j \exp x_j}$

Recall that $\text{softmax}(x) == \text{softmax}(x - \text{const})$. It makes possible to avoid computing exp() from large argument.
"""

class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input):
        """
        Computes the SoftMax activation for the given input.
        SoftMax is calculated as:
        softmax(x)_i = exp(x_i) / sum(exp(x_j))
        """
        # Numerical stability: subtract the maximum value per row
        stabilized_input = np.subtract(input, input.max(axis=1, keepdims=True))

        # Compute exponentials
        exp_input = np.exp(stabilized_input)

        # Compute SoftMax output
        self.output = exp_input / exp_input.sum(axis=1, keepdims=True)  # Normalize per row
        return self.output

    def updateGradInput(self, input, gradOutput):
        """
        Computes the gradient of the loss with respect to the input of the SoftMax layer.
        The gradient of SoftMax is a bit more involved:
        ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
        """
        # Initialize gradInput with the same shape as gradOutput
        self.gradInput = np.zeros_like(input)

        for i in range(input.shape[0]):  # Loop over the batch
            # Extract SoftMax output for the i-th sample
            sm = self.output[i].reshape(-1, 1)  # Column vector

            # Compute the Jacobian matrix of SoftMax
            jacobian = np.diagflat(sm) - sm @ sm.T

            # Multiply Jacobian by gradOutput for this sample
            self.gradInput[i] = gradOutput[i] @ jacobian

        return self.gradInput

    def __repr__(self):
        return "SoftMax"

"""## 3. LogSoftMax
- input:   **`batch_size x n_feats`**
- output: **`batch_size x n_feats`**

$\text{logsoftmax}(x)_i = \log\text{softmax}(x)_i = x_i - \log {\sum_j \exp x_j}$

The main goal of this layer is to be used in computation of log-likelihood loss.
"""

class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def updateOutput(self, input):
        """
        Computes the LogSoftMax activation for the given input.
        LogSoftMax is calculated as:
        logsoftmax(x)_i = x_i - log(sum(exp(x_j)))
        """
        # Numerical stability: subtract the maximum value per row
        stabilized_input = np.subtract(input, input.max(axis=1, keepdims=True))

        # Compute log-sum-exp
        log_sum_exp = np.log(np.sum(np.exp(stabilized_input), axis=1, keepdims=True))

        # Compute LogSoftMax
        self.output = stabilized_input - log_sum_exp
        return self.output

    def updateGradInput(self, input, gradOutput):
        """
        Computes the gradient of the loss with respect to the input of the LogSoftMax layer.
        The gradient is computed as:
        ∂L/∂x_i = gradOutput_i - sum(gradOutput * exp(output))
        """
        # Compute exp(LogSoftMax output)
        softmax_output = np.exp(self.output)

        # Compute sum of gradOutput * softmax_output across features
        grad_sum = np.sum(gradOutput, axis=1, keepdims=True)

        # Compute the gradient
        self.gradInput = gradOutput - softmax_output * grad_sum
        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"

"""## 4. Batch normalization
One of the most significant recent ideas that impacted NNs a lot is [**Batch normalization**](http://arxiv.org/abs/1502.03167). The idea is simple, yet effective: the features should be whitened ($mean = 0$, $std = 1$) all the way through NN. This improves the convergence for deep models letting it train them for days but not weeks. **You are** to implement the first part of the layer: features normalization. The second part (`ChannelwiseScaling` layer) is implemented below.

- input:   **`batch_size x n_feats`**
- output: **`batch_size x n_feats`**

The layer should work as follows. While training (`self.training == True`) it transforms input as $$y = \frac{x - \mu}  {\sqrt{\sigma + \epsilon}}$$
where $\mu$ and $\sigma$ - mean and variance of feature values in **batch** and $\epsilon$ is just a small number for numericall stability. Also during training, layer should maintain exponential moving average values for mean and variance:
```
    self.moving_mean = self.moving_mean * alpha + batch_mean * (1 - alpha)
    self.moving_variance = self.moving_variance * alpha + batch_variance * (1 - alpha)
```
During testing (`self.training == False`) the layer normalizes input using moving_mean and moving_variance.

Note that decomposition of batch normalization on normalization itself and channelwise scaling here is just a common **implementation** choice. In general "batch normalization" always assumes normalization + scaling.
"""

class BatchNormalization(Module):
    EPS = 1e-5  # Увеличено для большей численной стабильности

    def __init__(self, alpha=0.9):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha  # Momentum для экспоненциального сглаживания
        self.moving_mean = None  # Сглаженное среднее
        self.moving_variance = None  # Сглаженная дисперсия
        self.gamma = 1.0  # Масштабирующий параметр
        self.beta = 0.0  # Сдвиг
        self.gradGamma = None  # Градиент по gamma
        self.gradBeta = None  # Градиент по beta

    def updateOutput(self, input):
        """
        Прямой проход: нормализация входа и применение gamma и beta.
        """
        if self.training:
            # Вычисление среднего и дисперсии для текущего батча
            batch_mean = input.mean(axis=0, keepdims=True)
            batch_variance = input.var(axis=0, keepdims=True)

            # Обновление скользящих средних
            if self.moving_mean is None:
                self.moving_mean = batch_mean
                self.moving_variance = batch_variance
            else:
                self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * batch_mean
                self.moving_variance = self.alpha * self.moving_variance + (1 - self.alpha) * batch_variance

            # Нормализация входа
            self.normalized_input = (input - batch_mean) / np.sqrt(batch_variance + self.EPS)
        else:
            # Использование скользящих средних в режиме оценки
            self.normalized_input = (input - self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)

        # Применение gamma и beta
        self.output = self.gamma * self.normalized_input + self.beta
        return self.output

    def updateGradInput(self, input, gradOutput):
        """
        Обратный проход: вычисление градиентов по входу, gamma и beta.
        """
        batch_size = input.shape[0]
        batch_mean = input.mean(axis=0, keepdims=True)
        batch_variance = input.var(axis=0, keepdims=True)

        # Градиенты по gamma и beta
        self.gradGamma = np.sum(gradOutput * self.normalized_input, axis=0)
        self.gradBeta = np.sum(gradOutput, axis=0)

        # Градиенты по входу
        grad_norm = gradOutput * self.gamma
        grad_variance = np.sum(grad_norm * (input - batch_mean) * -0.5 * (batch_variance + self.EPS) ** -1.5, axis=0, keepdims=True)
        grad_mean = np.sum(grad_norm * -1 / np.sqrt(batch_variance + self.EPS), axis=0, keepdims=True) + \
                    grad_variance * np.mean(-2 * (input - batch_mean), axis=0, keepdims=True)

        self.gradInput = grad_norm / np.sqrt(batch_variance + self.EPS) + \
                         grad_variance * 2 * (input - batch_mean) / batch_size + \
                         grad_mean / batch_size
        return self.gradInput

    def zeroGradParameters(self):
        """
        Сбрасывает градиенты для gamma и beta.
        """
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def getParameters(self):
        """
        Возвращает параметры gamma и beta.
        """
        return [self.gamma, self.beta]

    def getGradParameters(self):
        """
        Возвращает градиенты по gamma и beta.
        """
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        """
        Читаемое представление слоя.
        """
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

"""Practical notes. If BatchNormalization is placed after a linear transformation layer (including dense layer, convolutions, channelwise scaling) that implements function like `y = weight * x + bias`, than bias adding become useless and could be omitted since its effect will be discarded while batch mean subtraction. If BatchNormalization (followed by `ChannelwiseScaling`) is placed before a layer that propagates scale (including ReLU, LeakyReLU) followed by any linear transformation layer than parameter `gamma` in `ChannelwiseScaling` could be freezed since it could be absorbed into the linear transformation layer.

## 5. Dropout
Implement [**dropout**](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). The idea and implementation is really simple: just multimply the input by $Bernoulli(p)$ mask. Here $p$ is probability of an element to be zeroed.

This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons.

While training (`self.training == True`) it should sample a mask on each iteration (for every batch), zero out elements and multiply elements by $1 / (1 - p)$. The latter is needed for keeping mean values of features close to mean values which will be in test mode. When testing this module should implement identity transform i.e. `self.output = input`.

- input:   **`batch_size x n_feats`**
- output: **`batch_size x n_feats`**
"""

class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p  # Вероятность "выключения" нейронов
        self.mask = None

    def updateOutput(self, input):
        """
        Applies dropout during training.
        """
        if self.training:
            # Генерируем маску с вероятностью "выключения" p
            self.mask = (np.random.rand(*input.shape) > self.p).astype(float)
            # Масштабируем маску, чтобы сохранить масштаб ожидания
            self.output = input * self.mask / (1.0 - self.p)
        else:
            # В режиме оценки Dropout не применяется
            self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        """
        Backward pass for Dropout.
        Applies the mask to the gradients during backpropagation.
        """
        if self.training:
            # Градиенты масштабируются так же, как и входы
            self.gradInput = gradOutput * self.mask / (1.0 - self.p)
        else:
            # Градиенты проходят без изменений в режиме оценки
            self.gradInput = gradOutput
        return self.gradInput

    def __repr__(self):
        return f"Dropout(p={self.p})"

"""# Activation functions

Here's the complete example for the **Rectified Linear Unit** non-linearity (aka **ReLU**):
"""

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

"""## 6. Leaky ReLU
Implement [**Leaky Rectified Linear Unit**](http://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29%23Leaky_ReLUs). Expriment with slope.
"""

class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super(LeakyReLU, self).__init__()
        self.slope = slope  # Коэффициент наклона для отрицательных значений

    def updateOutput(self, input):
        """
        Прямой проход через LeakyReLU:
        output_i = input_i, если input_i > 0
        output_i = slope * input_i, если input_i <= 0
        """
        self.output = np.where(input > 0, input, self.slope * input)
        return self.output

    def updateGradInput(self, input, gradOutput):
        """
        Обратный проход через LeakyReLU:
        gradInput_i = gradOutput_i, если input_i > 0
        gradInput_i = slope * gradOutput_i, если input_i <= 0
        """
        self.gradInput = gradOutput * np.where(input > 0, 1, self.slope)
        return self.gradInput

    def __repr__(self):
        return f"LeakyReLU(slope={self.slope})"

"""## 7. ELU
Implement [**Exponential Linear Units**](http://arxiv.org/abs/1511.07289) activations.
"""

class ELU(Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha  # Коэффициент для управления величиной выходного значения для отрицательных входов

    def updateOutput(self, input):
        """
        Прямой проход через ELU:
        output_i = input_i, если input_i > 0
        output_i = alpha * (exp(input_i) - 1), если input_i <= 0
        """
        self.output = np.where(input > 0, input, self.alpha * (np.exp(input) - 1))
        return self.output

    def updateGradInput(self, input, gradOutput):
        """
        Обратный проход через ELU:
        gradInput_i = gradOutput_i, если input_i > 0
        gradInput_i = gradOutput_i * alpha * exp(input_i), если input_i <= 0
        """
        elu_derivative = np.where(input > 0, 1, self.alpha * np.exp(input))
        self.gradInput = gradOutput * elu_derivative
        return self.gradInput

    def __repr__(self):
        return f"ELU(alpha={self.alpha})"

"""## 8. SoftPlus
Implement [**SoftPlus**](https://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29) activations. Look, how they look a lot like ReLU.
"""

class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, input):
        """
        Прямой проход через SoftPlus:
        output_i = log(1 + exp(input_i))
        """
        self.output = np.log1p(np.exp(input))  # Используем log1p для численной стабильности
        return self.output

    def updateGradInput(self, input, gradOutput):
        """
        Обратный проход через SoftPlus:
        gradInput_i = gradOutput_i * sigmoid(input_i)
        где sigmoid(input_i) = 1 / (1 + exp(-input_i))
        """
        sigmoid = 1 / (1 + np.exp(-input))  # Вычисляем сигмоиду
        self.gradInput = gradOutput * sigmoid
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"

"""# Criterions

Criterions are used to score the models answers.
"""

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

"""The **MSECriterion**, which is basic L2 norm usually used for regression, is implemented here for you.
- input:   **`batch_size x n_feats`**
- target: **`batch_size x n_feats`**
- output: **scalar**
"""

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

"""## 9. Negative LogLikelihood criterion (numerically unstable)
You task is to implement the **ClassNLLCriterion**. It should implement [multiclass log loss](http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss). Nevertheless there is a sum over `y` (target) in that formula,
remember that targets are one-hot encoded. This fact simplifies the computations a lot. Note, that criterions are the only places, where you divide by batch size. Also there is a small hack with adding small number to probabilities to avoid computing log(0).
- input:   **`batch_size x n_feats`** - probabilities
- target: **`batch_size x n_feats`** - one-hot representation of ground truth
- output: **scalar**


"""

class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15  # Small value for numerical stability

    def updateOutput(self, input, target):
        """
        Computes the Negative Log-Likelihood (NLL):
        output = -log(input[target])
        """
        # Ensure target is an array of indices
        if len(target.shape) > 1:  # If target is one-hot encoded
            target = np.argmax(target, axis=1)

        # Clamp input values to avoid log(0)
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        # Compute NLL loss
        self.output = -np.sum(np.log(input_clamp[np.arange(target.shape[0]), target])) / target.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        """
        Computes the gradient of NLL:
        gradInput[target] = -1 / input[target]
        """
        if len(target.shape) > 1:  # If target is one-hot encoded
            target = np.argmax(target, axis=1)

        # Clamp input values to avoid division by zero
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        # Initialize gradient input
        self.gradInput = np.zeros_like(input)
        self.gradInput[np.arange(target.shape[0]), target] = -1 / input_clamp[np.arange(target.shape[0]), target]
        self.gradInput /= target.shape[0]  # Normalize by batch size
        return self.gradInput





"""## 10. Negative LogLikelihood criterion (numerically stable)
- input:   **`batch_size x n_feats`** - log probabilities
- target: **`batch_size x n_feats`** - one-hot representation of ground truth
- output: **scalar**

Task is similar to the previous one, but now the criterion input is the output of log-softmax layer. This decomposition allows us to avoid problems with computation of forward and backward of log().
"""

class ClassNLLCriterion(Criterion):
    """
    Negative Log Likelihood Criterion (numerically stable).

    Input: batch_size x n_classes - log probabilities (output of LogSoftmax)
    Target: batch_size x n_classes - one-hot representation of ground truth
    Output: scalar loss value
    """

    def updateOutput(self, input, target):
        # Convert one-hot target to indices if necessary
        if target.ndim > 1:
            target = np.argmax(target, axis=1)
        
        # Compute the negative log likelihood loss
        # Loss = - (1 / batch_size) * sum_i input[i, target[i]]
        batch_size = input.shape[0]
        self.output = -np.mean(input[np.arange(batch_size), target])
        return self.output

    def updateGradInput(self, input, target):
        # Gradient of the loss w.r.t. input
        # gradInput[i, target[i]] = -1 / batch_size
        if target.ndim > 1:
            target = np.argmax(target, axis=1)
        
        batch_size = input.shape[0]
        self.gradInput = np.zeros_like(input)
        self.gradInput[np.arange(batch_size), target] = -1 / batch_size
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterion"


    


"""# Optimizers

### SGD optimizer with momentum
- `variables` - list of lists of variables (one list per layer)
- `gradients` - list of lists of current gradients (same structure as for `variables`, one array for each var)
- `config` - dict with optimization parameters (`learning_rate` and `momentum`)
- `state` - dict with optimizator state (used to save accumulated gradients)
"""

def sgd_momentum(variables, gradients, config, state):
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('accumulated_grads', {})

    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):

            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))

            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)

            current_var -= old_grad
            var_index += 1

"""## 11. [Adam](https://arxiv.org/pdf/1412.6980.pdf) optimizer
- `variables` - list of lists of variables (one list per layer)
- `gradients` - list of lists of current gradients (same structure as for `variables`, one array for each var)
- `config` - dict with optimization parameters (`learning_rate`, `beta1`, `beta2`, `epsilon`)
- `state` - dict with optimizator state (used to save 1st and 2nd moment for vars)

Formulas for optimizer:

Current step learning rate: $$\text{lr}_t = \text{learning_rate} * \frac{\sqrt{1-\beta_2^t}} {1-\beta_1^t}$$
First moment of var: $$\mu_t = \beta_1 * \mu_{t-1} + (1 - \beta_1)*g$$
Second moment of var: $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2)*g*g$$
New values of var: $$\text{variable} = \text{variable} - \text{lr}_t * \frac{m_t}{\sqrt{v_t} + \epsilon}$$
"""

def adam_optimizer(variables, gradients, config, state):
    """
    Реализация Adam-оптимизатора.

    Args:
        variables: список переменных модели (например, веса и смещения)
        gradients: список градиентов для переменных
        config: словарь с гиперпараметрами ('learning_rate', 'beta1', 'beta2', 'epsilon')
        state: словарь для хранения моментов и шага времени

    Returns:
        Обновляет переменные модели на месте.
    """
    # Инициализация моментов и шага времени
    state.setdefault('m', {})  # Первый момент
    state.setdefault('v', {})  # Второй момент
    state.setdefault('t', 0)   # Шаг времени
    state['t'] += 1  # Увеличиваем счетчик шага времени

    # Проверяем наличие всех необходимых гиперпараметров
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()

    # Вычисляем скорректированную скорость обучения
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2']**state['t']) / (1 - config['beta1']**state['t'])

    # Индекс переменной для отслеживания моментов
    var_index = 0

    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            # Получаем моменты для текущей переменной
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))

            # Обновляем первый момент
            np.add(config['beta1'] * var_first_moment, (1 - config['beta1']) * current_grad, out=var_first_moment)

            # Обновляем второй момент
            np.add(config['beta2'] * var_second_moment, (1 - config['beta2']) * (current_grad ** 2), out=var_second_moment)

            # Обновляем переменную
            current_var -= lr_t * var_first_moment / (np.sqrt(var_second_moment) + config['epsilon'])

            # Проверяем, что моменты обновлены
            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1

"""# Layers for advanced track homework
You **don't need** to implement it if you are working on `homework_main-basic.ipynb`

## 12. Conv2d [Advanced]
- input:   **`batch_size x in_channels x h x w`**
- output: **`batch_size x out_channels x h x w`**

You should implement something like pytorch `Conv2d` layer with `stride=1` and zero-padding outside of image using `scipy.signal.correlate` function.

Practical notes:
- While the layer name is "convolution", the most of neural network frameworks (including tensorflow and pytorch) implement operation that is called [correlation](https://en.wikipedia.org/wiki/Cross-correlation#Cross-correlation_of_deterministic_signals) in signal processing theory. So **don't use** `scipy.signal.convolve` since it implements [convolution](https://en.wikipedia.org/wiki/Convolution#Discrete_convolution) in terms of signal processing.
- It may be convenient to use `skimage.util.pad` for zero-padding.
- It's rather ok to implement convolution over 4d array using 2 nested loops: one over batch size dimension and another one over output filters dimension
- Having troubles with understanding how to implement the layer?
 - Check the last year video of lecture 3 (starting from ~1:14:20)
 - May the google be with you
"""

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

"""## 13. MaxPool2d [Advanced]
- input:   **`batch_size x n_input_channels x h x w`**
- output: **`batch_size x n_output_channels x h // kern_size x w // kern_size`**

You are to implement simplified version of pytorch `MaxPool2d` layer with stride = kernel_size. Please note, that it's not a common case that stride = kernel_size: in AlexNet and ResNet kernel_size for max-pooling was set to 3, while stride was set to 2. We introduce this restriction to make implementation simplier.

Practical notes:
- During forward pass what you need to do is just to reshape the input tensor to `[n, c, h / kern_size, kern_size, w / kern_size, kern_size]`, swap two axes and take maximums over the last two dimensions. Reshape + axes swap is sometimes called space-to-batch transform.
- During backward pass you need to place the gradients in positions of maximal values taken during the forward pass
- In real frameworks the indices of maximums are stored in memory during the forward pass. It is cheaper than to keep the layer input in memory and recompute the maximums.
"""

class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.gradInput = None

    def updateOutput(self, input):
        """
        Прямой проход для MaxPool2d:
        Подразумевается, что шаг равен размеру ядра.
        """
        batch_size, channels, input_h, input_w = input.shape
        kernel_size = self.kernel_size

        # Размеры выходного изображения
        output_h = input_h // kernel_size
        output_w = input_w // kernel_size

        # Инициализируем выход и индексы максимумов
        self.output = np.zeros((batch_size, channels, output_h, output_w))
        self.max_indices = np.zeros((batch_size, channels, output_h, output_w, 2), dtype=int)

        # Выполняем максимизацию в каждом окне
        for i in range(output_h):
            for j in range(output_w):
                # Координаты текущего окна
                start_i, start_j = i * kernel_size, j * kernel_size
                end_i, end_j = start_i + kernel_size, start_j + kernel_size

                # Извлекаем окна и находим максимумы
                window = input[:, :, start_i:end_i, start_j:end_j]
                max_vals = np.max(window, axis=(2, 3))
                max_indices = np.argmax(window.reshape(batch_size, channels, -1), axis=-1)

                # Сохраняем максимум и его индекс
                self.output[:, :, i, j] = max_vals
                max_indices_2d = np.unravel_index(max_indices, (kernel_size, kernel_size))
                self.max_indices[:, :, i, j, 0] = max_indices_2d[0] + start_i
                self.max_indices[:, :, i, j, 1] = max_indices_2d[1] + start_j

        return self.output

    def updateGradInput(self, input, gradOutput):
        """
        Обратный проход для MaxPool2d:
        Градиенты направляются только к тем элементам, которые были максимумами.
        """
        self.gradInput = np.zeros_like(input)
        batch_size, channels, output_h, output_w = gradOutput.shape
        kernel_size = self.kernel_size

        # Распределяем градиенты по максимумам
        for i in range(output_h):
            for j in range(output_w):
                for b in range(batch_size):
                    for c in range(channels):
                        max_i, max_j = self.max_indices[b, c, i, j]
                        self.gradInput[b, c, max_i, max_j] += gradOutput[b, c, i, j]

        return self.gradInput

    def __repr__(self):
        """
        Читаемое представление слоя.
        """
        q = f'MaxPool2d, kern {self.kernel_size}, stride {self.kernel_size}'
        return q

"""### Flatten layer
Just reshapes inputs and gradients. It's usually used as proxy layer between Conv2d and Linear.
"""

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
