# A General‑Purpose JAX-Based Controller

Learning controllers (classic PID or a small neural net) to regulate different dynamical systems (“plants”) using automatic differentiation. The framework is built on **JAX** and **NumPy**, with plotting via **Matplotlib**.

>
> Part of the NTNU Course IT3105.

---

##  Highlights

- **Two controller families**
  - **PIDController** (learns \(k_p, k_d, k_i\))
  - **NeuralNetController** (fully‑connected MLP with configurable layers & activations)
- **Pluggable plants**
  - Bathtub water level  
  - Cournot duopoly profit  
  - Drone velocity under quadratic drag
- **End‑to‑end differentiable**: backpropagates through plant dynamics with JAX.
- **Noise‑robust**: trains with configurable disturbance ranges.
- **Diagnostics**: training curves and (for PID) parameter histories are saved as PNGs.


## How it works 

- Each epoch rolls the plant for `timesteps` steps.  
- The controller receives features (tracking error, error delta, integral term, etc.) and outputs an action \(U\).  
- The plant’s `process(U, D, state, internal_variables)` advances the state under a random disturbance \(D\).  
- The objective is MSE between plant output and `controller_target`.  
- **Gradients** of MSE w.r.t. controller parameters are computed with JAX and updated via vanilla gradient descent.

---

## ➕ Extending the framework

### Add a new plant
Create a subclass of `Plant` in `plant.py`:
```python
class MyPlant(Plant):
    def __init__(self, init_state, ...):
        super().__init__(init_state)
        # store constants

    def initialize(self):
        self.state = self.state
        self.internal_variables = None  # or a dict/array

    def process(self, U, D, state, internal_variables):
        # return next_state, next_internal_variables
        return next_state, internal_variables
```



See the full write‑up with parameter tables, problem descriptions, and figures: **IT3105_Project_1_Report.pdf**.
