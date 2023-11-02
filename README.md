# Layered Control Architectures for Robotics

To install the required libraries run "pip install -r requirements.txt"
Once the required libraries are installed, to run the full pipeline use "python run_exp.py"

To use JAX with the GPU, please follow instructions from the official page linked below to install the right version of JAX. 
https://jax.readthedocs.io/en/latest/installation.html

The codes used in the paper for unicycle model experiments are found in the Simulations folder. It is organized as shown below.

Simulations/

* data/ - Data for the unicycle dynamical system
* mlp_jax.py - An implementation of the multilayer perceptron network using JAX libraries
* model_learning.py - An implementation of value function learning pipeline using JAX libraries
* helper_functions.py - Useful functions for computing tracking cost and input for the unicycle system
* generate_data.py - Contains the ILQR modules and unicycle simulation function and other helper functions
* run_exp.py - Example usage for running a full scale experiment on a simple unicycle system
* Testing_mlp.ipynb - Contains an example for generating trajectories for the unicycle system and visualizing them

To generate data, please take a look at how to call the functions as shown in the Testing_mlp.ipynb notebook. The generated data is
saved to the data/ folder. All codes from the PyTorch implementation are in the tracking/ folder.



