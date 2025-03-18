import json
import sys
import os

def generate_perf_table(perf_data_list):
    table_rows = []
    for experiment_name, perf_data in perf_data_list:
        row = f"""    ["{experiment_name}"], 
    [{perf_data["dTLB-loads"]["value"]}], 
    [{perf_data["dTLB-load-misses"]["value"]}], 
    [{perf_data["dTLB-load-misses"]["percentage"]:.2f}], 
    [{perf_data["page-faults"]["value"]}]"""
        table_rows.append(row)
    
    table = f"""#figure(
  text(size: 9pt)[
    #table(
    columns: (auto, auto, auto, auto, auto),
    inset: 6pt,
    align: horizon,
    table.header(
    [*Experiment*], [*dTLB-loads*], [*dTLB-load-misses*], [*dTLB-load-misses (%)*], [*page-faults*]
    ),
    {',\n'.join(table_rows)}
  )
  ],
  caption: [`perf` analysis of XGBoost training time and different optimisation techniques],
)
"""
    return table

def generate_strace_table(strace_data_list):
    table_rows = []
    for experiment_name, strace_data in strace_data_list:
        row = f"""    ["{experiment_name}"], 
    [{strace_data["mmap"]["calls"]}], 
    [{strace_data["brk"]["calls"]}], 
    [{strace_data["munmap"]["calls"]}]"""
        table_rows.append(row)
    
    table = f"""#figure(
  text(size: 9pt)[
    #table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: horizon,
    table.header(
    [*Experiment*], [*mmap (count)*], [*brk (count)*], [*munmap (count)*], 
    ),
    {',\n'.join(table_rows)}
  )
  ],
  caption: [`strace` analysis of XGBoost training time and different optimisation techniques],
)
"""
    return table

def process_folder(folder_path):
    strace_data_list = []
    perf_data_list = []
    
    for experiment_folder in os.listdir(folder_path):
        experiment_path = os.path.join(folder_path, experiment_folder)
        if os.path.isdir(experiment_path):
            strace_file = os.path.join(experiment_path, "run_1.strace.json")
            perf_file = os.path.join(experiment_path, "run_1.perf.json")
            
            if os.path.exists(strace_file) and os.path.exists(perf_file):
                with open(strace_file, 'r') as f:
                    strace_data = json.load(f)
                    strace_data_list.append((experiment_folder, strace_data))
                
                with open(perf_file, 'r') as f:
                    perf_data = json.load(f)
                    perf_data_list.append((experiment_folder, perf_data))
    
    return strace_data_list, perf_data_list

def main(folder_path, output_file):
    strace_data_list, perf_data_list = process_folder(folder_path)
    
    strace_table = generate_strace_table(strace_data_list)
    perf_table = generate_perf_table(perf_data_list)
    
    with open(output_file, 'w') as f:
        f.write(strace_table)
        f.write("\n\n")  # Add some space between tables
        f.write(perf_table)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <folder_path> <output_file>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    output_file = sys.argv[2]
    
    main(folder_path, output_file)
