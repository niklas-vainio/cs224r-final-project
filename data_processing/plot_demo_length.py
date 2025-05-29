# Visualization script to plot demo length
import os
import argparse
import h5py
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    # Get demo lengths
    demo_lengths = []

    with h5py.File(args.file, "r") as hf:
        for name, group in hf.items():
            num_steps = group["action"]["mobile_base"].shape[0]

            demo_lengths.append((int(name.split("_")[1]), num_steps))

    # Sort in order and plot
    demo_lengths.sort(key = lambda x: x[0])
    lengths_raw = [item[1] for item in demo_lengths]

    plt.bar(list(range(len(demo_lengths))), lengths_raw)
    plt.grid(True)
    plt.xlabel("Episode Number")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.savefig("plot.png")