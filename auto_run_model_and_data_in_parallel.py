#!/usr/bin/env python3
import os
import subprocess
import time
import traceback
from typing import List, Tuple, Union


# Execute this from "mtp" folder

output_dir = "./amplandocteract_files/others_011 (valid, 1 core, 4hr, baron)"
output_data_dir = f"{output_dir}/data"
CPU_CORES_PER_SOLVER = 1
MAX_PARALLEL_SOLVERS = 8
EXECUTION_TIME_LIMIT = (4 * 60 * 60) + (1 * 60) + 0  # Seconds, set this to any value <= 0 to ignore this parameter
MIN_FREE_RAM = 2  # GiB
MIN_FREE_SWAP = 8  # GiB, will only be used if "MIN_FREE_RAM == 0"

# NOTE: Use "double quotes ONLY" in the below variables three
# engine_path = './octeract-engine-4.0.0/bin/octeract-engine'
# engine_options = f'options octeract_options "num_cores={CPU_CORES_PER_SOLVER}";'
# process_name_to_stop_using_ctrl_c = 'mpirun' if CPU_CORES_PER_SOLVER > 1 else 'octeract-engine'
engine_path = './ampl.linux-intel64/baron'
engine_options = f'option baron_options "maxtime={EXECUTION_TIME_LIMIT - 60} threads={CPU_CORES_PER_SOLVER} barstats keepsol lsolmsg outlev=1 prfreq=100 prtime=2 problem";'
process_name_to_stop_using_ctrl_c = 'baron'

models_dir = "./Files/Models"
model_to_input_mapping = {
	"m1_basic.R"				: "./Files/Data/m1_m2",  # q
	"m2_basic2_v2.R"			: "./Files/Data/m1_m2",  # q1, q2
	"m3_descrete_segment.R"		: "./Files/Data/m3_m4",  # q
	"m4_parallel_links.R"		: "./Files/Data/m3_m4",  # q1, q2
}
data_files = [
	'd1_Sample_input_cycle_twoloop.dat',
	'd2_Sample_input_cycle_hanoi.dat',
	'd3_Sample_input_double_hanoi.dat',
	'd4_Sample_input_triple_hanoi.dat',
	'd5_Taichung_input.dat',
	'd6_HG_SP_1_4.dat',
	'd7_HG_SP_2_3.dat',
	'd8_HG_SP_3_4.dat',
	'd9_HG_SP_4_2.dat',
	'd10_HG_SP_5_5.dat',
	'd11_HG_SP_6_3.dat',
]


def run_command(cmd: str, debug_print: bool = False) -> Tuple[bool, str]:
	# REFER: Context-Search-fms
	if debug_print:
		print(f'DEBUG: COMMAND: `{cmd}`')
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
		return True, output
	except Exception as e:
		print(f'EXCEPTION OCCURRED (cmd=`{cmd}`), will return "0" as the output')
		# print(e)
		# print(traceback.format_exc())
	if debug_print:
		print("0")
	return False, "0"


def run_command_get_output(cmd: str, debug_print: bool = False) -> str:
	return run_command(cmd, debug_print)[1]


def get_free_ram() -> float:
	'''returns: free RAM in GiB'''
	# REFER: https://stackoverflow.com/questions/34937580/get-available-memory-in-gb-using-single-bash-shell-command/34938001
	return float(run_command_get_output(r'''awk '/MemFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo'''))


def get_free_swap() -> float:
	'''returns: free Swap in GiB'''
	# REFER: https://stackoverflow.com/questions/34937580/get-available-memory-in-gb-using-single-bash-shell-command/34938001
	return float(run_command_get_output(r'''awk '/SwapFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo'''))


def get_execution_time(pid: Union[int, str]) -> int:
	'''returns: execution time in seconds'''
	# REFER: https://unix.stackexchange.com/questions/7870/how-to-check-how-long-a-process-has-been-running
	success, output = run_command(f'ps -o etimes= -p "{pid}"')
	if success:
		return int(output)
	return 10**15  # ~3.17 crore years


def time_memory_monitor_and_stopper(
		execution_time_limit: float,
		min_free_ram: float,
		pids_to_monitor: List[str],
		pids_finished: List[str],
		blocking: bool
	) -> None:
	'''
	execution_time_limit: in seconds and ignored if <= 0
	min_free_ram        : in GiB
	blocking            : waiting until one of the PID in pids_to_monitor is stopped
	'''
	global CPU_CORES_PER_SOLVER, process_name_to_stop_using_ctrl_c
	to_run_the_loop = True
	while to_run_the_loop:
		to_run_the_loop = blocking
		if execution_time_limit > 0:
			for i_bashpid in pids_to_monitor:
				if get_execution_time(i_bashpid) >= execution_time_limit:
					# NOTE: only SIGINT signal does proper termination of the octeract-engine
					print(run_command_get_output("pstree -ap " + str(i_bashpid), debug_print=True))
					print(run_command_get_output("pstree -ap " + str(i_bashpid) + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # Time"))
					print(run_command_get_output("pstree -aps " + str(i_bashpid) + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # Time"))
					success, pid = run_command("pstree -ap " + str(i_bashpid) + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+' | grep -oE '[0-9]+'  # Time Monitor", debug_print=True)
					pids_finished.append(i_bashpid)
					to_run_the_loop = False
					if success:
						print(run_command_get_output(f'kill -s SIGINT {pid}  # Time Monitor', True))
					else:
						print(f'DEBUG: TIME_LIMIT: tmux session (with bash PID={i_bashpid}) already finished')
					time.sleep(2)
			for i_bashpid in pids_finished:
				pids_to_monitor.remove(i_bashpid)
			pids_finished.clear()
		if get_free_ram() <= min_free_ram:
			# Kill the oldest executing octeract instance used to solve data+model combination
			bashpid_tokill = sorted([(get_execution_time(p), p) for p in pids_to_monitor], reverse=True)[0][1]
			print(run_command_get_output("pstree -ap " + str(bashpid_tokill) + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # RAM"))
			print(run_command_get_output("pstree -aps " + str(bashpid_tokill) + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # RAM"))
			success, pid = run_command("pstree -ap " + str(bashpid_tokill) + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+' | grep -oE '[0-9]+'  # RAM Monitor", debug_print=True)
			pids_to_monitor.remove(bashpid_tokill)
			if success:
				print(run_command_get_output(f'kill -s SIGINT {pid}  # RAM Monitor', True))
			else:
				print(f'DEBUG: RAM_USAGE: tmux session (with bash PID={bashpid_tokill}) already finished')
			time.sleep(2)
			break
		time.sleep(2)


run_command_get_output(f'mkdir -p "{output_dir}"')
run_command_get_output(f'mkdir -p "{output_data_dir}"')
tmuxbashpids_to_monitor = list()
tmuxbashpids_finished = list()
for model_name, data_path_prefix in model_to_input_mapping.items():
	for ith_data_file in data_files:
		print(run_command_get_output('tmux ls | grep "autorun_"'))
		print(run_command_get_output('tmux ls | grep "autorun_" | wc -l'))
		while int(run_command_get_output('tmux ls | grep "autorun_" | wc -l')) >= MAX_PARALLEL_SOLVERS:
			print("----------")
			print(tmuxbashpids_to_monitor)
			print(tmuxbashpids_finished)
			time_memory_monitor_and_stopper(EXECUTION_TIME_LIMIT, MIN_FREE_RAM, tmuxbashpids_to_monitor, tmuxbashpids_finished, True)
			print(tmuxbashpids_to_monitor)
			print(tmuxbashpids_finished)
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
		run_command_get_output(
			rf'''
tmux new-session -d -s 'autorun_{short_uniq_combination}' 'echo $$ > /tmp/{short_uniq_combination}.txt ; ./ampl.linux-intel64/ampl > "{output_dir}/{short_uniq_combination}.txt" 2>&1 <<EOF
	reset;
	model {models_dir}/{model_name}
	data {data_path_prefix}/{ith_data_file}
	option solver "{engine_path}";
	{engine_options}
	solve;
	display _total_solve_time;
	display l;
	display {"q1,q2" if (short_model_name in ("m2", "m4")) else "q"};
EOF'
			'''
		)
		tmuxbashpid = run_command_get_output(f'cat "/tmp/{short_uniq_combination}.txt"')
		tmuxbashpids_to_monitor.append(tmuxbashpid)
		print(f'DEBUG: tmux session "{short_uniq_combination}" -> {tmuxbashpid}')
		time.sleep(2)
		# Copy files from /tmp folder at regular intervals to avoid losing data when system deletes them automatically
		run_command_get_output(f'cp -r /tmp/at*nl /tmp/at*octsol /tmp/baron_tmp* "{output_data_dir}"')

while len(tmuxbashpids_to_monitor) > 0:
	print(tmuxbashpids_to_monitor)
	print(tmuxbashpids_finished)
	time_memory_monitor_and_stopper(EXECUTION_TIME_LIMIT, MIN_FREE_RAM, tmuxbashpids_to_monitor, tmuxbashpids_finished, False)
	run_command_get_output(f'cp -r /tmp/at*nl /tmp/at*octsol /tmp/baron_tmp* "{output_data_dir}"')
	time.sleep(10)

time.sleep(60)
run_command_get_output(f'cp -r /tmp/at*nl /tmp/at*octsol /tmp/baron_tmp* "{output_data_dir}"')
