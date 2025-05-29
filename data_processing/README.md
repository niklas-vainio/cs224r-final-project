# Data Processing

This folder contains scripts for playback and exporting of data.

To expoort observations from raw demo files, run:
```
./export_all_data.sh {path_to_folder_with_hdf5_files}
```

To merge all `*_replay.hdf5` files in a directory (and subdirectories), run the following:
```
python merge_and_post_process_data.py --dir {path_to_directory} --output {output_file}
```

The other scripts can be used for viewing and debugging the contents of .hdf5 files.