#!/bin/bash

# 모든 하위 디렉토리를 포함하여 재귀적으로 .mp4 파일을 찾습니다.
find . -type f -name "*.mp4" | while read file; do
  # 출력 파일 경로를 생성합니다. (변경된 프레임 속도를 저장할 경로)
  output_dir=$(dirname "$file")
  output_file="$output_dir/converted_$(basename "$file")"
  
  # ffmpeg 명령어를 사용하여 프레임 속도를 변경합니다.
  ffmpeg -i "$file" -r 20 "$output_file"
done

