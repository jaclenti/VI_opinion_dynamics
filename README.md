# Variational Inference of Parameters in Opinion Dynamics Models
The repo contains the notebooks and scripts used to run the experiments of the paper ``Variational Inference of Parameters in Opinion Dynamics Models''.

The four opinion dynamics models analysed are:
- BCM-S. Bounded Confidence Model with Structural rule. *_Leader refer to this.
- BCM-I. BCM with Interaction rule. *_Feed refer to this.
- BCM-U. BCM with Update rule. *_isBackfire refer to this.
- BCM-G. BCM with Graph rule. *_Rewire refer to this.

For each experiment we have only the .pkl file containing the performance measures.
When running the experiments, the .npy files containing the opinion dynamics traces are also saved.

For running a single experiment it is enough to run the command $python src/inference_*.py input1 input2 ...

For running multiple parallel experiments, $bash src/run_experiments_*.sh can be used.
For using .sh commands, it is necessary to install GNU parallel.

