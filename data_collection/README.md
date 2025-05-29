# Data Collection

This folder contains scripts for launchind data collection. Requires a calibrated 7-DOF JoyLo device. Some configurations are hardcoded in the shell scripts, change then if needed.

For data collection, in one terminal run:
```
./launch_nodes.sh {output_path.hdf5}
```

and in another run:
```
./run_joylo.sh
```
