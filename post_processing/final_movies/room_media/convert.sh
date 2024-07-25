#!/bin/bash
for file in *.webm; do
    ffmpeg -i "$file" -c:v libx264 -an "${file%.webm}.mp4"
done

