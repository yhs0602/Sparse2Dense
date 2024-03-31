import re


def split_log_entries(log_line):
    # 알파벳 문자열과 시간 스탬프 사이를 분리하는 정규 표현식
    pattern = r"([a-zA-Z])(\d{2}:\d{2}:\d{2}\.\d{6})"
    # 알파벳과 시간 스탬프 사이에 구분자 삽입
    modified_log_line = re.sub(pattern, r"\1|\2", log_line)
    # 구분자를 기준으로 로그 라인 분리
    split_lines = modified_log_line.split("|")
    return split_lines


def fix_mixed_lines(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    fixed_lines = []
    num_fixed_lines = 0
    for line in lines:
        # 로그 라인을 정규 표현식을 사용해 분리
        split_lines = split_log_entries(line)
        if len(split_lines) > 1:
            num_fixed_lines += 1
        fixed_lines.extend(split_lines)

    print(f"Fixed {num_fixed_lines} mixed log lines.")
    # 분리된 로그 라인을 새 파일에 저장
    with open("fixed_" + file_path, "w") as file:
        for line in fixed_lines:
            file.write(line.strip() + "\n")
    print(f"Fixed log file saved as fixed_{file_path}.")


if __name__ == "__main__":
    # 로그 파일 경로를 지정
    log_file_path = "java_log_filtered.csv"
    fix_mixed_lines(log_file_path)
