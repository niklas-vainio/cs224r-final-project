# Data Collection

This folder contains scripts for launchind data collection. Requires a calibrated 7-DOF JoyLo device. Some configurations are hardcoded in the shell scripts, change them if needed.

For data collection, in one terminal run:
```
./launch_nodes.sh {output_path.hdf5}
```

and in another run:
```
./run_joylo.sh
```

**Refer to the installation instructions in `og-gello` to configure an environment with OmniGibson and its dependencies**