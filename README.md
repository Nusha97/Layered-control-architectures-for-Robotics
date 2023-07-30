# Layered-architecture-quadrotor-control

Simulations/

* data/ - Data for the unicycle dynamical system
* mlp_jax.py - An implementation of the multilayer perceptron network using JAX libraries
* model_learning.py - An implementation of value function learning pipeline using JAX libraries
* helper_functions.py - Useful functions for computing tracking cost and input for the unicycle system
* generate_data.py - Contains the ILQR modules and unicycle simulation function and other helper functions
* run_exp.py - Example usage for running a full scale experiment on a simple unicycle system
* Testing_mlp.ipynb - Contains an example for generating trajectories for the unicycle system and visualizing them

To run the full pipeline, use python run_exp.py after installing all the requirements



