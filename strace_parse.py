import json
import re
import argparse

def extract_strace_output(full_output):
    strace_start = full_output.find("% time     seconds  usecs/call     calls    errors syscall")
    if strace_start == -1:
        raise ValueError("Strace output not found in the input")
    return full_output[strace_start:]

def parse_strace_output(output):
    lines = output.strip().split('\n')
    data = {}
    
    # Parse syscall data
    for line in lines[2:-2]:  # Skip header and summary lines
        fields = line.split()
        if len(fields) == 6:
            syscall = fields[5]
            data[syscall] = {
                "percent_time": float(fields[0]),
                "seconds": float(fields[1]),
                "usecs_per_call": int(fields[2]),
                "calls": int(fields[3]),
                "errors": int(fields[4]) if fields[4] != "-" else 0
            }
        elif len(fields) == 5:
            syscall = fields[4]
            data[syscall] = {
                "percent_time": float(fields[0]),
                "seconds": float(fields[1]),
                "usecs_per_call": int(fields[2]),
                "calls": int(fields[3]),
                "errors": 0
            }
    
    # Parse summary
    summary_match = re.search(r'(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+)', lines[-1])
    if summary_match:
        data["summary"] = {
            "percent_time": float(summary_match.group(1)),
            "seconds": float(summary_match.group(2)),
            "usecs_per_call": int(summary_match.group(3)),
            "calls": int(summary_match.group(4))
        }
    
    return data

def format_float(f):
    return f'{f:.6f}'

def save_as_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse strace output and save as JSON")
    parser.add_argument("--input_file", help="Path to the input strace file")
    parser.add_argument("--output_file", help="Path to the output JSON file")
    args = parser.parse_args()

    # Read strace output from input file
    with open(args.input_file, 'r') as f:
        full_output = f.read()
    
    # Extract strace part
    strace_output = extract_strace_output(full_output)
    
    parsed_data = parse_strace_output(strace_output)
    save_as_json(parsed_data, args.output_file)
    
    print(f"Strace results saved to {args.output_file}")
