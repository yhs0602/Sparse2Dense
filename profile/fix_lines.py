import re


def split_log_entries(log_line):
    # Regular expression to separate between alphabetic strings and timestamps
    pattern = r"([a-zA-Z])(\d{2}:\d{2}:\d{2}\.\d{6})"
    # Insert a separator between alphabets and timestamps
    modified_log_line = re.sub(pattern, r"\1|\2", log_line)
    # Separating log lines by delimiter
    split_lines = modified_log_line.split("|")
    return split_lines


def fix_mixed_lines(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    fixed_lines = []
    num_fixed_lines = 0
    for line in lines:
        # Separate log lines using a regular expression
        split_lines = split_log_entries(line)
        if len(split_lines) > 1:
            num_fixed_lines += 1
        fixed_lines.extend(split_lines)

    print(f"Fixed {num_fixed_lines} mixed log lines.")
    # Save the separated log lines to a new file
    with open("fixed_" + file_path, "w") as file:
        for line in fixed_lines:
            file.write(line.strip() + "\n")
    print(f"Fixed log file saved as fixed_{file_path}.")


if __name__ == "__main__":
    # Specify the path to the log file
    log_file_path = "java_log_filtered.csv"
    fix_mixed_lines(log_file_path)
