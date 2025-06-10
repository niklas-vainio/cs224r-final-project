# Training Robotics Policies With Imitation Learning from Simulated Teleoperation: A Proof of Concept for the BEHAVIOR-1k Project

## Directory Structure
* `data_collection`: Scripts for gathering teleoperation data with OmniGibson and JoyLo
* `data_processing`: Scripts for exporting observations and processing data into BRS-compatible format
* `train`: Scripts for training the WB-VIMA policy
* `eval`: Scripts for evaluating the WB-VIMA policy in the Omnigibson environment
* `external`: Pinned clones of [og-gello](https://github.com/StanfordVL/og-gello) and [brs-algo](https://github.com/behavior-robot-suite/brs-algo), which this repo depends on

`data_collection`, `data_processing`, and `eval` are intended to be run on any machine with OmniGibson installed. `train` requires a powerful GPU and does not depend on OmniGibson.

**Refer to each sub-directory's README file for more information.**

Niklas Vainio (niklasv)  
CS 224R Final project  
Spring Quarter 2025
