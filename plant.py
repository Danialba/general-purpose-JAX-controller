import jax.numpy as jnp
import numpy as np
import jax

class Plant:
    def __init__(self, state, internal_variables=None):
        # Common plant initialization goes here
        self.state=state
        
class Bathtub(Plant):
    def __init__(self, state, a, c, gravity):
        super().__init__(state)
        self.a=a
        self.c=c
        self.gravity=gravity
        
    def initialize(self):
        self.state=self.state
        self.internal_variables=None
        
    def velocity(self,H):
        if H<=0:
            return 0
        return jnp.sqrt(2*self.gravity*H)
    
    def flowrate(self,V,C):
        return V*C
    
    def change_bathtub_volume(self,change, Q):
        return change - Q
    
    def change_water_height(self,change_bathtub_volume, A):
        return change_bathtub_volume/A
    
        
    def process(self, U, D, state, internal_variables):
        velocity=self.velocity(state)
        flowrate=self.flowrate(velocity,self.c)
        change_bathtub_volume=self.change_bathtub_volume(U+D,flowrate)
        change_water_height=self.change_water_height(change_bathtub_volume,self.a)
        state+=change_water_height
        return state, internal_variables
        
       
class CournotCompetition(Plant):
    def __init__(self, state, max_price, marginal_cost, q1, q2):
        super().__init__(state)
        self.max_price=max_price
        self.marginal_cost=marginal_cost
        self.q1=q1
        self.q2=q2
    
    def initialize(self):
        self.state=self.state
        self.internal_variables=[self.q1, self.q2]
          
    
    def process(self, U, D, state, internal_variables):
        internal_variables[0]= 1/(1+jnp.exp(-(internal_variables[0] + U)))
        internal_variables[1]=1/(1+jnp.exp(-(internal_variables[1] + D)))
        q=internal_variables[0]+internal_variables[1]
        price=self.max_price-q
        profit=internal_variables[0]*(price-self.marginal_cost)
        state=profit
        return state, internal_variables
                
 
class Drone(Plant):
    
    def __init__(self, state, drag, mass):
        super().__init__(state)
        self.drag=drag
        self.mass=mass
        
    
    def initialize(self):
        self.state=self.state
        self.internal_variables=None
    
    def process(self, U, D, state, internal_variables):
        state+=((-self.drag*D*state**2)+U)/self.mass
        return state, internal_variables   
    
