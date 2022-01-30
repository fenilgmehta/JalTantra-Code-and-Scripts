#!/usr/bin/env python3
import os
import subprocess
import time
import traceback


def run_command_get_output(cmd: str, debug_print: bool = False) -> str:
	# REFER: Context-Search-fms
	if debug_print:
		print(f'DEBUG: COMMAND: {cmd}')
	try:
		# NOTE: Not using the below line of code because "sh" shell does not seem to properly parse the command
		#       Example: `kill -s SIGINT 12345`
		#                did not work and gave the following error:
		#                '/bin/sh: 1: kill: invalid signal number or name: SIGINT'
		#       The error logs of testing has been put in "REPO/logs/2022-01-22_ssh_kill_errors.txt"
		# status_code, output = subprocess.getstatusoutput(cmd)
		output = subprocess.check_output(['/usr/bin/bash', '-c', cmd], stderr=subprocess.STDOUT, shell=False).decode().strip()
		if debug_print:
			print(output)
		return output
	except Exception as e:
		print(f'EXCEPTION OCCURRED (cmd="{cmd}"), will return "0" as the output')
		# print(e)
		# print(traceback.format_exc())
	if debug_print:
		print("0")
	return "0"

def get_free_ram() -> float:
	'''returns: free RAM in GiB'''
	# REFER: https://stackoverflow.com/questions/34937580/get-available-memory-in-gb-using-single-bash-shell-command/34938001
	return float(run_command_get_output(r'''awk '/MemFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo'''))

def get_execution_time(pid) -> int:
	'''returns: execution time in seconds'''
	# REFER: https://unix.stackexchange.com/questions/7870/how-to-check-how-long-a-process-has-been-running
	return int(run_command_get_output(f'ps -o etimes= -p "{pid}"').strip())

def time_memory_monitor_and_stopper(execution_time_limit, min_free_ram, pids_to_monitor, pids_finished, blocking):
	'''
	execution_time_limit: in hours and ignored if <= 0
	min_free_ram        : in GiB
	blocking            : waiting until one of the PID in pids_to_monitor is stopped
	'''
	to_run_the_loop = blocking
	while to_run_the_loop:
		if execution_time_limit > 0:
			for i_bashpid in pids_to_monitor:
				if get_execution_time(i_bashpid) >= (execution_time_limit * 60 * 60):
					# NOTE: only SIGINT signal does proper termination of the octeract-engine
					pid = run_command_get_output(r"pstree -aps " + str(i_bashpid) + r" | grep -oE 'mpirun,[0-9]+' | grep -oE '[0-9]+'")
					print(run_command_get_output(f'kill -s SIGINT {pid}', True))
					pids_finished.append(i_bashpid)
					to_run_the_loop = False
					time.sleep(2)
			for i_bashpid in pids_finished:
				pids_to_monitor.remove(i_bashpid)
			pids_finished.clear()
		if get_free_ram() <= min_free_ram:
			# Kill the oldest executing octeract instance used to solve data+model combination
			bashpid_tokill = sorted([(get_execution_time(p), p) for p in pids_to_monitor], reverse=True)[0][1]
			pid = run_command_get_output(r"pstree -aps " + str(bashpid_tokill) + r" | grep -oE 'mpirun,[0-9]+' | grep -oE '[0-9]+'")
			print(run_command_get_output(f'kill -s SIGINT {pid}', True))
			pids_to_monitor.remove(bashpid_tokill)
			time.sleep(2)
			break
		time.sleep(2)


# Execute this from "mtp" folder
engine_path = "./octeract-engine-4.0.0/bin/octeract-engine"
models_dir = "./Files/Models"
model_to_input_mapping = {
	"m1_basic.mod"				: "./Files/Data/m1_m2",  # q
	"m2_basic2.mod"				: "./Files/Data/m1_m2",  # q1, q2
	"m3_descrete_segment.mod"	: "./Files/Data/m3_m4",  # q
	"m4_parallel_links.mod"		: "./Files/Data/m3_m4",  # q1, q2
}
data_files = [
	'd1_Sample_input_cycle_twoloop.dat',
	'd2_Sample_input_cycle_hanoi.dat',
	'd3_Sample_input_double_hanoi.dat',
	'd4_Sample_input_triple_hanoi.dat',
	# 'd5_Taichung_input.dat',
	# 'd6_HG_SP_1_4.dat',
	# 'd7_HG_SP_2_3.dat',
	# 'd8_HG_SP_3_4.dat',
	# 'd9_HG_SP_4_2.dat',
	# 'd10_HG_SP_5_5.dat',
	# 'd11_HG_SP_6_3.dat',
]

output_dir = "./amplandocteract_files/others"
output_data_dir = "./amplandocteract_files/others/data"
run_command_get_output(f'mkdir -p {output_dir}')
run_command_get_output(f'mkdir -p {output_data_dir}')
PID_ABOVE = 1197086
EXECUTION_TIME_LIMIT = 20 / 60 / 60  # Hours, set this to any value <= 0 to ignore this parameter
MIN_FREE_RAM = 2  # GiB

pids_to_monitor = list()
pids_finished = list()
for model_name, data_path_prefix in model_to_input_mapping.items():
	for ith_data_file in data_files:
		print(run_command_get_output('tmux ls | grep "autorun_"'))
		print(run_command_get_output('tmux ls | grep "autorun_" | wc -l'))
		while int(run_command_get_output('tmux ls | grep "autorun_" | wc -l')) >= 3:
			time_memory_monitor_and_stopper(EXECUTION_TIME_LIMIT, MIN_FREE_RAM, pids_to_monitor, pids_finished, True)
			# if EXECUTION_TIME_LIMIT > 0 or get_free_ram() <= 2:
			# 	print('Please kill some processes, Time limit exceeded or Low on memory')
			# 	print('Free RAM =', get_free_ram())
			# 	print(run_command_get_output('date'))
			# 	# # RAM is <= 1.5 GiB, so, send ctrl+c to a running AMPL program
			# 	# pid = int(run_command_get_output(''))
			# 	# # # TMUX_SERVER_PID = int(run_command_get_output("ps -e | grep 'tmux: server' | awk '{print $1}'"))  # 4573 <- manually found this
			# 	# # # run_command_get_output(f"pstree -aps {TMUX_SERVER_PID} | grep 'ampl,' | grep -o -E '[0-9]+' | sort -n | tail -n 3")
			# 	# for pid in run_command_get_output(r"ps -e | grep mpirun | grep -v grep | awk '{print $1}' | sort -n").split():
			# 	for pid in run_command_get_output(r"pstree -aps " + str(os.getpid()) + r" | grep -oE 'mpirun,[0-9]+' | grep -oE '[0-9]+' | sort -n", True).split():
			# 		# REFER: https://bash.cyberciti.biz/guide/Sending_signal_to_Processes
			# 		if not (int(pid) > PID_ABOVE):
			# 			print(f'DEBUG: not killing PID={pid} as it is <= {PID_ABOVE} (threshold)')
			# 			continue
			# 		# NOTE: only SIGINT signal does proper termination of the octeract-engine
			# 		print(run_command_get_output(f'kill -s SIGINT {pid}', True))
			# 	print()
			# 	break
			# time.sleep(100)
			# REFER: https://stackoverflow.com/a/66771847
		short_model_name = model_name[:model_name.find('_')]
		short_data_file_name = ith_data_file[:ith_data_file.find('_')]
		short_uniq_combination = f'{short_model_name}_{short_data_file_name}'
		print(short_model_name, short_data_file_name, short_uniq_combination)
		bashpid = run_command_get_output(
			rf'''
tmux new-session -d -s 'autorun_{short_uniq_combination}' './ampl.linux-intel64/ampl > "{output_dir}/{short_uniq_combination}.txt" 2>&1 <<EOF
	reset;
	model {models_dir}/{model_name}
	data {data_path_prefix}/{ith_data_file}
	option solver "{engine_path}";
	options octeract_options "num_cores=16";
	solve;
	display _total_solve_time;
	display l;
	display {"q1,q2" if (short_model_name in ("m2", "m4")) else "q"};
EOF'
			'''
		)
		bashpid = bashpid.split()[0]
		pids_to_monitor.append(bashpid)
		time.sleep(2)
		# Copy files from /tmp folder at regular intervals to avoid losing data when system deletes them automatically
		run_command_get_output(f'cp /tmp/at*nl /tmp/at*octsol "{output_data_dir}"')

while len(pids_to_monitor) > 0:
	time_memory_monitor_and_stopper(EXECUTION_TIME_LIMIT, MIN_FREE_RAM, pids_to_monitor, pids_finished, False)
	run_command_get_output(f'cp /tmp/at*nl /tmp/at*octsol "{output_data_dir}"')
	time.sleep(10)

run_command_get_output(f'cp /tmp/at*nl /tmp/at*octsol "{output_data_dir}"')
