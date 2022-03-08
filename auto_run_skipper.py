#!/usr/bin/env python3
import os
import subprocess
import time
import traceback
from typing import Tuple


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


n = 0
while int(run_command_get_output('tmux ls | grep "autorun_m2" | wc -l')) == 0:
    n += 1
    print(n, '', end='', flush=True)
    time.sleep(300)
    if n == 10:
        n = 0
        print('\r', end='', flush=True)

while int(run_command_get_output('tmux ls | grep -E "autorun_m(3|4)" | wc -l')) != 3:
    for j in ["d5", "d6", "d7", "d8", "d9", "d10", "d11", ]:
        run_command_get_output(f'tmux kill-session -t autorun_m2_{j}')
        time.sleep(20)
