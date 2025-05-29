#!/bin/bash

# Print out files
echo "This script will extract observations from the following files:"
for FILE in "$1"/*.hdf5; do
    echo "    * $FILE"
done
echo "(This will take a while!)"

# Ask for confirmation
echo ""
read -p "Do you wish to proceed? (y/n)" answer
if [[ "$answer" == [Yy] ]]; then
    echo "Proceeding..."
    sleep 1
else
    echo "Aborting!"
    exit 1
fi

# Run export on each file
for FILE in $1/*.hdf5; do
    clear
    echo "Exporting $FILE"
    sleep 5
    python scripts/replay_observations.py --files $FILE >> EXPORT_LOG
done

