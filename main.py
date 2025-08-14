from controller import PIDController
from controller import NeuralNetController
from consys import CONSYS
from plant import Bathtub
from plant import CournotCompetition
from plant import Drone


consys_epochs=100 #number of epochs
consys_timesteps=50 #number of timesteps
consys_learning_rate=0.00001 #learning rate
consys_disturbance_range=(0.9, 1.1) #range of disturbance

controller=NeuralNetController #PIDController or NeuralNetController
controller_target= 10.0 #Target value for the controller
pid_initial_params=[0.0, 0.0, 2.0] # [PIDController]: k_p, k_d, k_i - initial parameters for the PID controller
nn_initial_range_weights=(-0.1,0.1) # [Neuralnetcontroller]: range of initial weights
nn_initial_range_biases=(-0.1,0.1) # [Neuralnetcontroller]: range of initial biases
nn_controller_layers=(3,1) # [Neuralnetcontroller]: number of nodes in each layer 
nn_controller_activation_functions=("relu",) # [Neuralnetcontroller], tuple: activation functions for each non-input layer. Options: relu, tanh, sigmoid, linear
 

plant=Drone #Bathtub, CournotCompetition or drone

bathtub_init_height=20.0 #initial height of water in bathtub
bathtub_cross_sectional_area=10.0 #cross sectional area of bathtub
bathtub_drain_cross_sectional_area=0.1 #cross sectional area of bathtub drain
bathtub_gravity=9.8 #gravity

cournot_init_q1=0.5 #initial quantity of q1
cournot_init_q2=0.5 #initial quantity of q2
cournot_init_profit=50.0 #initial profit
cournot_max_price=100.0 #maximum price
cournot_marginal_cost=1.0 #marginal cost

drone_init_velocity=10.0 #initial velocity of drone
drone_mass=20.0 #mass of drone
drone_drag_coefficient=1.5 #drag coefficient


def main():
    
    if controller==PIDController:
        controller1=controller(controller_target, pid_initial_params)
    elif controller==NeuralNetController:
        controller1=controller(controller_target, nn_controller_layers, nn_controller_activation_functions, nn_initial_range_weights, nn_initial_range_biases)
        
    if plant==Bathtub:
        plant1=plant(bathtub_init_height, bathtub_cross_sectional_area, bathtub_drain_cross_sectional_area, bathtub_gravity)
    elif plant==CournotCompetition:
        plant1=plant(cournot_init_profit, cournot_max_price, cournot_marginal_cost, cournot_init_q1, cournot_init_q2)
    elif plant==Drone:
        plant1=plant(drone_init_velocity, drone_drag_coefficient, drone_mass)
        
    consys1=CONSYS(controller1, plant1, consys_epochs, consys_timesteps, consys_learning_rate, consys_disturbance_range)
    consys1.run_system()
    
    
if __name__ == "__main__":
    main()
