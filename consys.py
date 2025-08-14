import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib
from controller import PIDController


class CONSYS:

    def __init__(self, controller, plant, epochs, timesteps, learning_rate, disturbance_range):
        self.controller = controller
        self.plant = plant
        self.epochs=epochs
        self.timesteps=timesteps
        self.learning_rate=learning_rate
        self.disturbance_range=disturbance_range
        self.mse_history=[] # for plotting
        self.param_history=[] # for plotting
        self.plant_state_history=[] #for plotting
            
            
    def run_system(self):
        
        #Generate initial parameters for the controller
        self.controller.gen_params()
        for _ in range(self.epochs):
            
            #Reset error history and plant state
            self.controller.reset_current_features()
            self.plant.initialize()
            
            mse, gradients = jax.value_and_grad(self.run_one_epoch, argnums=0)(self.controller.params, self.timesteps, self.controller.target, self.controller.current_features, self.plant.state, self.plant.internal_variables)            
            print(f' Epoch: {_} - MSE: {mse} ')
            
            
            self.param_history.append(self.controller.params)
            self.mse_history.append(mse)
            #Update controller-parameters
            self.controller.update_params(self.controller.params, gradients, self.learning_rate)
            
    
        self.plot_mse_history(self.mse_history)
        if isinstance(self.controller, PIDController):
            self.plot_Ks(self.param_history)

        
        
    def run_one_epoch(self, params, timesteps, target, current_features, current_state, current_internal_variables):
        disturbance= np.random.uniform(self.disturbance_range[0],self.disturbance_range[1] , size=timesteps)
        
        cumulative_squared_error = 0
        prev_error = None
        
        for timestep in range(timesteps):
            error= target - current_state
            current_features[0]=error
            if prev_error is None:
                current_features[1]=0
            else:
                current_features[1]=error-prev_error
            current_features[2]+=error
            #jax.debug.print("Current features: {} ", current_features)
            U=self.controller.predict(params, current_features)
            #Generating controller output and updating plant state based on output and random disturbance
            current_state, current_internal_variables= self.plant.process(U, disturbance[timestep], current_state, current_internal_variables)
            cumulative_squared_error+=error**2
            prev_error=error
            #jax.debug.print("State: {} " " Output: {}", current_state, U )
        jax.debug.print('Plant state: {}', current_state)
        mse = cumulative_squared_error/timesteps
        return mse
    
        
        
    def plot_mse_history(self, mse_history):
        matplotlib.rcParams["figure.dpi"] = 100
        x = np.linspace(0,len(mse_history) , len(mse_history)) 
        plt.plot(x, mse_history)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Progression')
        plt.savefig('mse_history.png', dpi=1200)
        plt.show()
            
        
    def plot_Ks(self, Ks):
        Ks=np.array(Ks)
        k_p = Ks[:, 0]
        k_d = Ks[:, 1]
        k_i = Ks[:, 2]

        plt.plot(k_p, label='k_p')
        plt.plot(k_d, label='k_d')
        plt.plot(k_i, label='k_i')

        plt.xlabel('Epochs')
        plt.ylabel('Values')
        plt.title(' Control parameters')
               

        plt.legend()
        plt.grid(True)
        plt.savefig('parameter_history.png', dpi=1200)
        plt.show()
                

    
