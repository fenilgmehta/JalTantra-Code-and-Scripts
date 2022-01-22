#!/usr/bin/env python3
import os
import subprocess
import time
import traceback


def run_command_get_output(cmd: str) -> str:
	# REFER: Context-Search-fms
	try:
		# NOTE: Not using the below line of code because "sh" shell does not seem to properly parse the command
		#       Example: `kill -s SIGINT 12345`
		#                did not work and gave the following error:
		#                '/bin/sh: 1: kill: invalid signal number or name: SIGINT'
		#       The error logs of testing has been put in "REPO/logs/2022-01-22_ssh_kill_errors.txt"
		# status_code, output = subprocess.getstatusoutput(cmd)
		output = subprocess.check_output(['/usr/bin/bash', '-c', cmd], stderr=subprocess.STDOUT, shell=False).decode().strip()
		return output
	except Exception as e:
		print(e)
		print(traceback.format_exc())
	return "0"


# Execute this from "mtp" folder
engine_path = "./octeract-engine-4.0.0/bin/octeract-engine"
models_dir = "./Files/Models"
model_to_input_mapping = {
	"basic.mod"				: "./Files/Data/Basic",
	"basic2.mod"			: "./Files/Data/Basic",
	"descrete_segment.mod"	: "./Files/Data",
	"parallel_links.mod"	: "./Files/Data",
}
data_files = [
	'Sample_input_cycle_twoloop.dat',
	'Sample_input_cycle_hanoi.dat',
	'Sample_input_double_hanoi.dat',
	'Sample_input_triple_hanoi.dat',
	# 'HG_SP_1_4.dat',
	# 'HG_SP_2_3.dat',
	# 'HG_SP_3_4.dat',
	# 'HG_SP_4_2.dat',
	# 'HG_SP_5_5.dat',
	# 'HG_SP_6_3.dat',
	# 'Taichung_input.dat',
]

output_dir = "./amplandocteract_files/others"
output_data_dir = "./amplandocteract_files/others/data"
os.system(f'mkdir -p {output_dir}')
os.system(f'mkdir -p {output_data_dir}')

for model_name, data_path_prefix in model_to_input_mapping.items():
	for ith_data_file in data_files:
		print(run_command_get_output('tmux ls | grep "autorun_" | wc -l'))
		while int(run_command_get_output('tmux ls | grep "autorun_" | wc -l')) >= 3:
			# REFER: https://stackoverflow.com/questions/34937580/get-available-memory-in-gb-using-single-bash-shell-command/34938001
			if float(run_command_get_output(r'''awk '/MemFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo''')) <= 2:
				print('Low on memory. Please kill some processes')
				print('Free RAM =', float(run_command_get_output(r'''awk '/MemFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo''')))
				print(run_command_get_output('date'))
				# # RAM is <= 1.5 GiB, so, send ctrl+c to a running AMPL program
				# pid = int(run_command_get_output(''))
				# # # TMUX_SERVER_PID = int(run_command_get_output("ps -e | grep 'tmux: server' | awk '{print $1}'"))  # 4573 <- manually found this
				# # # run_command_get_output(f"pstree -aps {TMUX_SERVER_PID} | grep 'ampl,' | grep -o -E '[0-9]+' | sort -n | tail -n 3")
				PID_ABOVE = 846813
				for pid in run_command_get_output(r"ps -e | grep mpirun | grep -v grep | awk '{print $1}' | sort -n").split():
					# REFER: https://bash.cyberciti.biz/guide/Sending_signal_to_Processes
					if not (int(pid) > PID_ABOVE):
						print(f'DEBUG: not killing PID={pid} as it is <= {PID_ABOVE} (threshold)')
						continue
					# NOTE: only SIGINT signal does proper termination of the octeract-engine
					print(run_command_get_output(f'kill -s SIGINT {pid}'))
				print()
				break
			time.sleep(100)
			# REFER: https://stackoverflow.com/a/66771847
		os.system(
			rf'''
tmux new-session -d -s 'autorun_{model_name[:-4]}_{ith_data_file[:-4]}' './ampl.linux-intel64/ampl > "{output_dir}/{model_name[:-4]}_{ith_data_file[:-4]}.txt" 2>&1 <<EOF
	model {models_dir}/{model_name}
	data {data_path_prefix}/{ith_data_file}
	option solver "{engine_path}";
	options octeract_options "num_cores=16";
	solve;
	display _total_solve_time;
	display l;
	display {"q1,q2" if ("basic2" in model_name or "parallel_links" in model_name) else "q"};
EOF'
			'''
		)
		time.sleep(2)
		os.system(f'cp /tmp/at*nl /tmp/at*octsol "{output_data_dir}"')

