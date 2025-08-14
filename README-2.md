# A General‑Purpose JAX-Based Controller

Learning controllers (classic PID or a small neural net) to regulate different dynamical systems (“plants”) using automatic differentiation. The framework is built on **JAX** and **NumPy**, with plotting via **Matplotlib**.

> Author: **Danial Bashir**  
> Course: IT3105 — Project 1

---

## ✨ Highlights

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

---

## 🧱 Repository structure

```
.
├── consys.py        # Training loop (epochs, MSE, plotting)
├── controller.py    # PIDController and NeuralNetController
├── plant.py         # Bathtub, CournotCompetition, Drone plants
├── main.py          # Entry point and configuration
├── IT3105_Project_1_Report.pdf  # Full write-up & results
└── (figures saved at runtime)
    ├── mse_history.png
    └── parameter_history.png       # only for PID runs
```

---

## 🚀 Quick start

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U "jax[cpu]" numpy matplotlib
# If you have GPU/TPU, install the matching JAX build per JAX docs.
```

### 2) Choose a controller & plant in `main.py`
At the top of `main.py`, set:
```python
controller = NeuralNetController   # or PIDController
plant      = Drone                 # or Bathtub, CournotCompetition
```
…then adjust the corresponding hyperparameters (see next section).

### 3) Run
```bash
python main.py
```
Training will print progress and save:
- `mse_history.png` (learning curve)
- `parameter_history.png` (PID only)

---

## ⚙️ Configuration cheat‑sheet

Global training knobs in `main.py`:
```python
consys_epochs = 100            # training epochs
consys_timesteps = 50          # rollout length per epoch
consys_learning_rate = 0.01    # gradient descent step size
consys_disturbance_range = (-0.01, 0.01)  # sampled each step
controller_target = 20.0       # desired plant output
```

### PID controller
```python
pid_initial_params = [k_p0, k_d0, k_i0]
```

### Neural‑net controller
```python
nn_initial_range_weights = (-0.1, 0.1)
nn_initial_range_biases  = (-0.1, 0.1)
nn_controller_layers     = [input, h1, h2, ..., output]  # e.g. [3, 10, 10, 10, 1]
nn_controller_activation_functions = ["tanh","tanh","tanh","linear"]
```

### Plants

**Bathtub**
```python
bathtub_init_height = 20.0
bathtub_cross_sectional_area = 10.0
bathtub_drain_cross_sectional_area = 0.1
bathtub_gravity = 9.8
```

**CournotCompetition**
```python
cournot_init_q1 = 0.5
cournot_init_q2 = 0.5
cournot_init_profit = 50.0
cournot_max_price = 100.0
cournot_marginal_cost = 1.0
```

**Drone**
```python
drone_init_velocity = 10.0
drone_mass = 20.0
drone_drag_coefficient = 1.5
```

---

## 📈 Sample results

The report contains full runs and commentary. A few example learning curves (PID vs NN):

- Bathtub MSE — PID vs NN  
  ![Bathtub PID](Bathtub_PID_MSE.png)  
  ![Bathtub NN](Bathtub_nn_MSE.png)

- Drone MSE — PID vs NN  
  ![Drone PID](Drone_PID_MSE.png)  
  ![Drone NN](Drone_nn_MSE.png)

You’ll also get a fresh `mse_history.png` each time you run, and for PID a `parameter_history.png` tracking \(k_p, k_d, k_i\).

For a deeper discussion of hyperparameters, stability, and the effect of disturbance ranges, see **IT3105_Project_1_Report.pdf**.

---

## 🧩 How it works (brief)

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

### Add a new controller
Create a subclass of `Controller` in `controller.py` that implements:
- parameter initialization,
- a `forward`/`step` mapping from features to control \(U\),
- a `loss` (MSE is provided in `CONSYS`), and
- `update_params` using gradients from JAX.

---

## 📦 Suggested `requirements.txt`

```txt
numpy
matplotlib
jax
jaxlib
```

> For GPUs/TPUs, pin JAX/JAXLIB to your CUDA/CUDNN toolchain as per the official JAX installation guide.

---

## 🔖 License

MIT (feel free to change if you prefer another license).

---

## 📚 Report

See the full write‑up with parameter tables, problem descriptions, and figures: **IT3105_Project_1_Report.pdf**.
