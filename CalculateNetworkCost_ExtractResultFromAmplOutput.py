#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Union

# REFER: https://stackoverflow.com/questions/3172470/actual-meaning-of-shell-true-in-subprocess
g_BASH_PATH = subprocess.check_output(['which', 'bash'], shell=False).decode().strip()


def run_command(cmd: str, default_result: str = '') -> Tuple[bool, str]:
    """
    `stderr` is merged with `stdout`

    Returns:
        Tuple of [ok, output]
    """
    # REFER: Context-Search-fms and CalculateNetworkCost.py for this function
    global g_BASH_PATH
    try:
        # NOTE: Not using the below line of code because "sh" shell does not seem to properly parse the command
        #       Example: `kill -s SIGINT 12345`
        #                did not work and gave the following error:
        #                '/bin/sh: 1: kill: invalid signal number or name: SIGINT'
        #       The error logs of testing has been put in "REPO/logs/2022-01-22_ssh_kill_errors.txt"
        # status_code, output = subprocess.getstatusoutput(cmd)
        output = subprocess.check_output(
            [g_BASH_PATH, '-c', cmd],
            stderr=subprocess.STDOUT,
            shell=False
        ).decode().strip()
        return True, output
    except subprocess.CalledProcessError as e:
        print('DEBUG: Some error occurred, e = ' + str(e), file=sys.stderr)
    return False, default_result


# ---

# Step 1: Validate and store command line arguments in appropriate variables
# print('DEBUG:', sys.argv, file=sys.stderr)
if len(sys.argv) <= 3:
    print('Usage: python3 CalculateNetworkCost_ExtractResultFromAmplOutput.py /path/to/std_out_err.txt '
          '/path/to/NetworkFile PIPE_LEN_THRESHOLD')
    exit(2)
IN_STD_OUT_ERR_FILE_PATH = sys.argv[1]
IN_NETWORK_FILE_PATH = sys.argv[2]
IN_ARC_LEN_ERROR_THRESHOLD = float(sys.argv[3])
if not os.path.isfile(IN_STD_OUT_ERR_FILE_PATH):
    print(f"ERROR: No such file: '{IN_STD_OUT_ERR_FILE_PATH}'")
    exit(1)
if not os.path.isfile(IN_NETWORK_FILE_PATH):
    print(f"ERROR: No such file: '{IN_NETWORK_FILE_PATH}'")
    exit(1)

# ---

# Step 2: Read the file and extract only the necessary section using `awk`
cmd = """cat '""" + IN_STD_OUT_ERR_FILE_PATH.replace("'", r"\'") + r"""' | awk '
/:\s*q1\s*q2\s*:=/ { f += 1; }
/q\s*:=/ { f += 1; }
/^_total_solve_time.*/ { f += f; }
{ if (f==1) print; }
/^_total_solve_time.*/ { f += 1; }
'
"""
ok, ampl_output = run_command(cmd)
if not ok:
    print(f'ERROR: `run_command` failed for {cmd=}', file=sys.stderr)
    print(f'DEBUG: {ampl_output=}', file=sys.stderr)
    exit(1)
del cmd, ok
# print('DEBUG:', f'{ampl_output=}', file=sys.stderr)

# Step 3: Start parsing the file content
arc_len_calculated: Dict[Tuple[int, int], List[List[Union[int, float]]]] = defaultdict(list)
# Split each table into a different "list element", and then iterate using `for` loop
for res_table in [i.strip() for i in ampl_output.strip().strip('l;').split('\n\n')]:
    lines = res_table.splitlines(keepends=False)
    line0 = lines[0].strip()
    # print('DEBUG:', f'{line0=}', file=sys.stderr)
    x = int(line0[line0.index('[') + 1:line0.index(',')])
    cols = [int(i) for i in lines[1].strip().strip(':= \t').split()]
    # print('DEBUG:', f'{lines=}', file=sys.stderr)
    for line in lines[2:]:
        line = line.split()
        # print('DEBUG:', f'{line=}', file=sys.stderr)
        pipe_id = int(line[0])
        line = [float(i) for i in line[1:]]
        for i in range(len(cols)):
            if line[i] <= 1e-5:  # NOTE: Important Assumption that pipe length decently high
                continue
            arc_len_calculated[(x, cols[i])].append([pipe_id, line[i], ])

del ampl_output, res_table

# ---

# NOTE: Not sure if the checking done in step 4 and step 5 is required or not. I think that to some extent
#       it is required because due to rounding errors in float numbers, things like `20.000001 + 80.000001`
#       can result in sum being not "exactly equal to" the expected arc length (arc is same as an edge in graph).
#       Search for "# Sample OUTPUT" in this file to see an example

# Step 4: NOTE: Read `arc_len_expected` from network file (`network_file_name`) which was
#               used to get ".../std_out_err.txt" and validate `arc_len_calculated`

ok, output = run_command("cat '{}'".format(IN_NETWORK_FILE_PATH.replace("'", r"\'")))
if not ok:
    print('ERROR: `run_command` failed for cmd:\n\t\t' +
          "cat '{}'".format(IN_NETWORK_FILE_PATH.replace("'", r"\'")), file=sys.stderr)
    print(f'DEBUG: {output=}', file=sys.stderr)
    exit(1)
network_file_data = output.splitlines(keepends=False)
del ok, output

arcs_table_start_flag = False
arc_len_expected: Dict[Tuple[int, int], float] = dict()
for line in network_file_data:
    line = line.strip()
    if arcs_table_start_flag:
        if len(line) > 2:
            cols = line.rstrip(';').split()
            arc_len_expected[(int(cols[0]), int(cols[1]),)] = float(cols[2])
        if ';' in line:
            break
    # NOTE: Blank lines will automatically get skipped, no action will be taken for them
    # print('DEBUG:', f'{line=}', file=sys.stderr)
    # REFER: https://stackoverflow.com/questions/9012008/pythons-re-return-true-if-string-contains-regex-pattern
    if type(line) is str and re.search(r'param\s*:\s*arcs\s*:\s*L', line):
        arcs_table_start_flag = True
        continue

del network_file_data, arcs_table_start_flag, line

# Step 5: Fix the rounding error issues due to floating point numbers
# Rows are "Pipe IDs", Columns are "Destination Vertex", (tr) in the output denotes transposed data
for arc, pipes in arc_len_calculated.items():
    pipe_len_sum = 0
    for pipe_id, pipe_len in pipes:
        pipe_len_sum += pipe_len
    if abs(arc_len_expected[arc] - pipe_len_sum) > IN_ARC_LEN_ERROR_THRESHOLD:
        print(f'ERROR with {arc=}, {pipes=}, {pipe_len_sum=}, {arc_len_expected[arc]=}', file=sys.stderr)
        print(f'ERROR: DEBUG: {arc_len_expected=}', file=sys.stderr)
        print(f'ERROR: DEBUG: {arc_len_calculated=}', file=sys.stderr)
        exit(1)
    if pipe_len_sum != arc_len_expected[arc]:
        print(f'INFO : FIXING: {arc=}, {pipes=}, {pipe_len_sum=}, {arc_len_expected[arc]=}', file=sys.stderr)
        arc_len_calculated[arc][-1][-1] += (arc_len_expected[arc] - pipe_len_sum)

del arc_len_expected, arc, pipes

# ---

# Step 6: Print the output in a format similar to Competitive Programming
#         question for further processing by the caller of this program
print('DEBUG:', arc_len_calculated, file=sys.stderr)
# Print -> NUMBER_OF_ARCS_ie_EDGES
print(len(arc_len_calculated))
for i_arc, j_pipes_list in arc_len_calculated.items():
    # Print -> ARC_SOURCE_VERTEX, ARC_DESTINATION_VERTEX, OPTIMAL_NUMBER_OF_PIPES_REQUIRED
    print(i_arc[0], i_arc[1], len(j_pipes_list))
    for pipe_id, pipe_len in j_pipes_list:
        # Print -> PIPE_ID, PIPE_LENGTH
        print(pipe_id, pipe_len)

# ---

"""
# Sample INPUT
NOTE: The below table was kept as a reference for developing the above parsing algorithm

output = '''

l [1,*,*] (tr)
:          2           :=
1       5.95336e-15
2       1.74215e-13
3       1.26041e-12
4       5.19053e-12
5       4.19193e-11
6       2.62281e-10
7       7.00798e-08
8    1000
9       4.54441e-09
10      4.57397e-10
11      2.00471e-10
12      1.27575e-10
13      5.79978e-11
14      2.82772e-11

 [2,*,*] (tr)
:          3               4           :=
1      1.6634e-11       8.84086e-15
2      6.3528e-10       2.58764e-13
3    131.508            1.87427e-12
4    868.492            7.7472e-12
5      3.31667e-09      6.44088e-11
6      1.24695e-09      4.55572e-10
7      6.88067e-10   1000
8      3.62467e-10      3.86832e-09
9      2.87006e-10      1.53246e-09
10     1.76652e-10      3.72091e-10
11     1.16781e-10      1.81086e-10
12     8.72199e-11      1.19197e-10
13     4.78525e-11      5.61723e-11
14     2.5617e-11       2.78319e-11

 [3,*,*] (tr)
:          5           :=
1    1000
2       6.67849e-09
3       2.72134e-09
4       1.70899e-09
5       1.05494e-09
6       6.869e-10
7       4.74199e-10
8       2.92841e-10
9       2.41523e-10
10      1.58301e-10
11      1.08468e-10
12      8.24978e-11
13      4.63955e-11
14      2.51935e-11

 [4,*,*] (tr)
:          5               6          :=
1      4.05865e-14     2.4098e-13
2      1.18997e-12     7.11781e-12
3      8.70334e-12     5.41934e-11
4      3.70924e-11     2.61915e-10
5      3.98921e-10   381.307
6    874.898         618.693
7    125.102           1.84166e-09
8      9.00581e-10     5.50649e-10
9      5.66979e-10     3.95346e-10
10     2.55785e-10     2.12725e-10
11     1.47122e-10     1.31567e-10
12     1.03181e-10     9.52234e-11
13     5.2302e-11      5.01676e-11
14     2.68411e-11     2.62662e-11

 [6,*,*] (tr)
:          7           :=
1    1000
2       6.75279e-09
3       2.73408e-09
4       1.71403e-09
5       1.05686e-09
6       6.87714e-10
7       4.74587e-10
8       2.92988e-10
9       2.41624e-10
10      1.58345e-10
11      1.08488e-10
12      8.25095e-11
13      4.63992e-11
14      2.51946e-11

 [7,*,*] (tr)
:          5           :=
1       1.96126e-13
2       5.78503e-12
3       4.37284e-11
4       2.06678e-10
5       1.54879e-08
6    1000
7       2.1446e-09
8       5.77433e-10
9       4.09431e-10
10      2.16793e-10
11      1.33122e-10
12      9.60379e-11
13      5.03932e-11
14      2.6328e-11
;

'''

# ---

# Sample OUTPUT
# NOTE: both of the below commands mean the same
# command: (dev) ➜  mtp python CalculateNetworkCost_ExtractResultFromAmplOutput.py '/home/student/VirtualBox VMs/VM_Desktop/mtp/NetworkResults/792961099240ae9ceef3d764afb6d3e8/octeract_m1_792961099240ae9ceef3d764afb6d3e8/std_out_err.txt' '/home/student/VirtualBox VMs/VM_Desktop/mtp/DataNetworkGraphInput_hashed/792961099240ae9ceef3d764afb6d3e8.R' 1
# command: (dev) ➜  mtp python CalculateNetworkCost_ExtractResultFromAmplOutput.py '/home/student/VirtualBox VMs/VM_Desktop/mtp/NetworkResults/792961099240ae9ceef3d764afb6d3e8/octeract_m1_792961099240ae9ceef3d764afb6d3e8/std_out_err.txt' '/home/student/VirtualBox VMs/VM_Desktop/mtp/NetworkResults/792961099240ae9ceef3d764afb6d3e8/0_graph_network_data_testcase.R' 1
INFO : FIXING: arc=(2, 4), pipes=[[7, 916.017], [8, 83.9832]], pipe_len_sum=1000.0002000000001, arc_len_expected[arc]=1000.0
INFO : FIXING: arc=(4, 6), pipes=[[4, 81.0637], [5, 918.936]], pipe_len_sum=999.9997000000001, arc_len_expected[arc]=1000.0
DEBUG: defaultdict(<class 'list'>, {(1, 2): [[8, 1000.0]], (2, 3): [[3, 137.676], [4, 862.324]], (2, 4): [[7, 916.017], [8, 83.98299999999993]], (3, 5): [[1, 1000.0]], (4, 6): [[4, 81.0637], [5, 918.9363]], (4, 5): [[7, 1000.0]], (6, 7): [[6, 1000.0]], (7, 5): [[6, 1000.0]]})
8
1 2 1
8 1000.0
2 3 2
3 137.676
4 862.324
2 4 2
7 916.017
8 83.98299999999993
3 5 1
1 1000.0
4 6 2
4 81.0637
5 918.9363
4 5 1
7 1000.0
6 7 1
6 1000.0
7 5 1
6 1000.0
"""
