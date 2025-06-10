# Training

**The scripts in this folder require a powerful GPU to run. For the project, they were run on a `g5.2xlarge` instance on AWS.**
**To set up an environment, follow installation instructions in the `brs-algo` repo**

This repository uses Weights and Biases for training metrics. To get started, create a new project, add its ID to `train_model.sh`, and then run `wandb login`.

To run model training, change the `DATA_ROOT_DIR` variable to point to your collected dataset and then run
```
./train_model.sh
```

Metrics can be viewed on the Weights and Biases dashboard.


