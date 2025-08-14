import numpy as np
import jax.numpy as jnp
import jax

class Controller:
    def __init__(self, target): 

        self.target=target
        self.params=None
        self.current_error=0
        self.prev_error=0
        self.current_features= np.array([0,0,0]) #error, error change, error sum
        
        @property
        def target(self):
            return self.target
        @property
        def params(self):
            return self.params
        @property
        def current_error(self):
            return self.current_error
        @property
        def prev_error(self):
            return self.prev_error
        @property
        def current_features(self):
            return self.current_features

    def reset_current_features(self):
        self.current_features[0]=0.0
        self.current_features[1]=0.0
        self.current_features[2]=0.0
        
    def update_current_features(self, prev_error, error):
        self.current_features[0]=error
        if prev_error is None:
            self.current_features[1]=0
        else:
            self.current_features[1]=error - prev_error
        self.current_features[2]+=error
        
        return self.current_features
        
class PIDController(Controller):
    def __init__(self, target, initial_params):
        super().__init__(target)
        self.initial_params=initial_params
 
    def gen_params(self):
        #TODO: hvordan skal man generere initielle parametre classic PID?
        self.params=np.array(self.initial_params)
        
    def predict(self, params, features):
        #jax.debug.print("U: {} ", params[0] * features[0] + params[1] * features[1] + params[2] * features[2])
        return params[0] * features[0] + params[1] * features[1] + params[2] * features[2]
    
    def update_params(self, params, gradients, learning_rate):
        new_params= np.array([params[0]-learning_rate*gradients[0], params[1]-learning_rate*gradients[1], params[2]-learning_rate*gradients[2]])
        self.params=new_params

class NeuralNetController(Controller):
    
    def __init__(self, target, layers, activation_functions, initial_range_weights, initial_range_biases):
        super().__init__(target)
        self.layers=layers
        self.initial_range_weights=initial_range_weights
        self.initial_range_biases=initial_range_biases
        self.activation_functions = [getattr(self, activation_function) if len(activation_functions) > 1 
                                     else getattr(self, activation_functions[0]) for activation_function in activation_functions]

    def gen_params(self): 
        sender = self.layers[0]; params = []
        for receiver in self.layers[1:]:
            weights = np.random.uniform(self.initial_range_weights[0], self.initial_range_weights[1],(sender,receiver)) 
            biases= np.random.uniform(self.initial_range_biases[0], self.initial_range_biases[1],(1,receiver))
            sender = receiver
            params.append([weights, biases])
        self.params=params
        
    def predict(self, all_params, features):
        activations = features
        for (weights, biases), activation_func in zip (all_params, self.activation_functions):
            activations = activation_func(jnp.dot(activations, weights) + biases)
        return activations[0][0]

    def update_params(self, params, gradients, learning_rate):
        self.params = [(w - learning_rate * dw, b - learning_rate * db)
                for (w, b), (dw, db) in zip(params, gradients)]
        
    def relu(self, x):
        return jax.nn.relu(x)

    def tanh(self, x):
        return jax.nn.tanh(x)
    
    def sigmoid(self, x):
        return jax.nn.sigmoid(x)
    
    def linear(self, x):
        return x
