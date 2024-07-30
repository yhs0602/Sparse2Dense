#!/bin/bash

# Output file
output_file="frame_counts.txt"

# Initialize the output file
echo "Filename, Frame Count" > $output_file

# Iterate over all mp4 files in the directory and subdirectories
find . -type f -name "*.mp4" | while read -r video_file; do
  # Get the frame count using ffprobe
  frame_count=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$video_file")
  
  # Check if frame_count is not empty
  if [ -n "$frame_count" ]; then
    # Append the filename and frame count to the output file
    echo "$(basename "$video_file"), $frame_count" >> $output_file
  else
    echo "Error processing $video_file"
  fi
done

echo "Frame count mapping completed. Check $output_file for results."

