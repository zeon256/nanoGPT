import sys
import json
import re

def parse_perf_output(output):
    data = {}
    
    # Regular expression to match perf stat output lines
    pattern = r'\s*([\d,]+)\s+(\S+)(?::u)?\s*(.+)?'
    
    lines = output.strip().split('\n')
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            value, event, description = match.groups()
            
            # Remove commas from the value and convert to integer
            value = int(value.replace(',', ''))
            
            # Remove ':u' suffix from event name if present
            event = event.replace(':u', '')
            
            # Extract percentage if present
            percentage_match = re.search(r'#\s*([\d.]+)%', description or '')
            percentage = float(percentage_match.group(1)) if percentage_match else None
            
            data[event] = {
                "value": value,
                "percentage": percentage,
                "description": description.strip() if description else None
            }
    
    # Extract time information
    time_elapsed = re.search(r'([\d.]+)\s+seconds time elapsed', output)
    user_time = re.search(r'([\d.]+)\s+seconds user', output)
    sys_time = re.search(r'([\d.]+)\s+seconds sys', output)
    
    if time_elapsed:
        data['time_elapsed'] = float(time_elapsed.group(1))
    if user_time:
        data['user_time'] = float(user_time.group(1))
    if sys_time:
        data['sys_time'] = float(sys_time.group(1))
    
    return data

def save_as_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_perf.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r') as f:
        perf_output = f.read()
    
    parsed_data = parse_perf_output(perf_output)
    save_as_json(parsed_data, output_file)
    
    print(f"Perf stat results saved to {output_file}")
