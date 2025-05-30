# This file is a simple util file to see the contents of a .hdf5 file

import h5py
import numpy as np
import argparse

def view_hdf5_file(filename: str, attr=None, field = None):
    assert filename.endswith(".hdf5")
    
    def print_item(name, obj):
        depth = name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{' ' * 2 * depth}{name:<70}{' ' * (10-2*depth)} {obj.shape} {obj.dtype}")
        else:
            print(f"{' ' * 2 * depth}{name}")
    
    with h5py.File(filename, 'r') as hf:
        if field:
            if attr:
                print(hf[field].attrs[attr])
            else:
                print(hf[field])
                print(f"Attributes: {hf[field].attrs.keys()}")
                print("Min: ", np.min(hf[field], axis=0))
                print("Max: ", np.max(hf[field], axis=0))
        else:
            hf.visititems(print_item)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--attr", required=False, default=None)
    parser.add_argument("--field", required=False, default=None)
    
    args = parser.parse_args()
    view_hdf5_file(args.file, args.attr, args.field)