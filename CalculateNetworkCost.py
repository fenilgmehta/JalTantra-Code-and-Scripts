#!/usr/bin/env python3
import argparse
import subprocess
import time
from typing import List, Tuple, Union, Dict

# ---

# NOTE
# Assumptions
#   1. Linux OS is used for execution
#   2. `bash`, `which`, `nproc`, `tmux` are installed
#   3. Execution is done from "mtp" folder

# ---

# REFER: https://stackoverflow.com/questions/3172470/actual-meaning-of-shell-true-in-subprocess
BASH_PATH = subprocess.check_output(['which', 'bash'], shell=False).decode().strip()


def run_command(cmd: str, default_result: str = '0', debug_print: bool = False) -> Tuple[bool, str]:
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
        output = subprocess.check_output(
            [BASH_PATH, '-c', cmd],
            stderr=subprocess.STDOUT,
            shell=False
        ).decode().strip()
        if debug_print:
            print(output)
        return True, output
    except Exception as e:
        print(f'EXCEPTION OCCURRED (cmd=`{cmd}`), will return default_result ("{default_result}") as the output')
    # print(e)
    # print(traceback.format_exc())
    if debug_print:
        print(default_result)
    return False, default_result


def run_command_get_output(cmd: str, debug_print: bool = False) -> str:
    return run_command(cmd, debug_print=debug_print)[1]


# ---

def get_free_ram() -> float:
    """returns: free RAM in GiB"""
    # REFER: https://stackoverflow.com/questions/34937580/get-available-memory-in-gb-using-single-bash-shell-command/34938001
    return float(run_command_get_output(r'''awk '/MemFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo'''))


def get_free_swap() -> float:
    """returns: free Swap in GiB"""
    # REFER: https://stackoverflow.com/questions/34937580/get-available-memory-in-gb-using-single-bash-shell-command/34938001
    return float(run_command_get_output(r'''awk '/SwapFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo'''))


def get_execution_time(pid: Union[int, str]) -> int:
    """returns: execution time in seconds"""
    # REFER: https://unix.stackexchange.com/questions/7870/how-to-check-how-long-a-process-has-been-running
    success, output = run_command(f'ps -o etimes= -p "{pid}"')
    if success:
        return int(output)
    return 10 ** 15  # ~3.17 crore years


class AutoExecutorSettings:
    AVAILABLE_SOLVERS = ['baron', 'octeract']
    AVAILABLE_MODELS = [1, 2, 3, 4]

    def __init__(self):
        self.output_dir = './NetworkResults/'.rstrip('/')  # Note: Do not put trailing forward slash ('/')
        self.output_data_dir = f'{self.output_dir}/SolutionData'
        self.CPU_CORES_PER_SOLVER = 1
        # 48 core server is being used
        self.MAX_PARALLEL_SOLVERS = 44
        # Time is in seconds, set this to any value <= 0 to ignore this parameter
        self.EXECUTION_TIME_LIMIT = (0 * 60 * 60) + (5 * 60) + 0
        self.MIN_FREE_RAM = 2  # GiB
        self.MIN_FREE_SWAP = 8  # GiB, usefulness of this variable depends on the swappiness of the system
        # NOTE: Update `AutoExecutorSettings.AVAILABLE_MODELS`
        self.models_dir = "./Files/Models"  # m1, m3 => q   ,   m2, m4 => q1, q2
        self.models: List = [None, 'm1_basic.R', 'm2_basic2_v2.R', 'm3_descrete_segment.R', 'm4_parallel_links.R']
        self.solvers: Dict = {}
        self.__update_solver_dict()

    def set_execution_time_limit(self, hours: int, minutes: int, seconds: int) -> None:
        self.EXECUTION_TIME_LIMIT = (hours * 60 * 60) + (minutes * 60) + seconds
        self.__update_solver_dict()

    def set_cpu_cores_per_solver(self, n: int) -> None:
        self.CPU_CORES_PER_SOLVER = n
        self.__update_solver_dict()

    def __update_solver_dict(self):
        # NOTE: Update `AutoExecutorSettings.AVAILABLE_SOLVERS` if keys in below dictionary are updated
        # NOTE: Use double quotes ONLY in the below variables
        self.solvers: Dict = {
            'baron': {
                'engine_path': './ampl.linux-intel64/baron',
                'engine_options': f'option baron_options "maxtime={self.EXECUTION_TIME_LIMIT - 10} '
                                  f'threads={self.CPU_CORES_PER_SOLVER} barstats keepsol lsolmsg '
                                  f'outlev=1 prfreq=100 prtime=2 problem";',
                'process_name_to_stop_using_ctrl_c': 'baron'
            },
            'octeract': {
                'engine_path': './octeract-engine-4.0.0/bin/octeract-engine',
                'engine_options': f'options octeract_options "num_cores={self.CPU_CORES_PER_SOLVER}";',
                'process_name_to_stop_using_ctrl_c': 'mpirun' if self.CPU_CORES_PER_SOLVER > 1 else 'octeract-engine'
            }
        }


g_auto_executor_settings = AutoExecutorSettings()


# ---

def time_memory_monitor_and_stopper(
        execution_time_limit: float,
        min_free_ram: float,
        pids_to_monitor: List[str],
        pids_finished: List[str],
        blocking: bool
) -> None:
    """
    execution_time_limit: in seconds and ignored if <= 0
    min_free_ram        : in GiB
    blocking            : waiting until one of the PID in pids_to_monitor is stopped
    """
    global CPU_CORES_PER_SOLVER, process_name_to_stop_using_ctrl_c
    to_run_the_loop = True
    while to_run_the_loop:
        to_run_the_loop = blocking
        if execution_time_limit > 0:
            for i_bashpid in pids_to_monitor:
                if get_execution_time(i_bashpid) >= execution_time_limit:
                    # NOTE: only SIGINT signal does proper termination of the octeract-engine
                    print(run_command_get_output("pstree -ap " + str(i_bashpid), debug_print=True))
                    print(run_command_get_output("pstree -ap " + str(i_bashpid)
                                                 + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # Time"))
                    print(run_command_get_output("pstree -aps " + str(i_bashpid)
                                                 + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # Time"))
                    success, pid = run_command("pstree -ap " + str(i_bashpid)
                                               + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+' | "
                                                 f"grep -oE '[0-9]+'  # Time Monitor",
                                               debug_print=True)
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
            print(run_command_get_output("pstree -ap " + str(bashpid_tokill)
                                         + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # RAM"))
            print(run_command_get_output("pstree -aps " + str(bashpid_tokill)
                                         + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # RAM"))
            success, pid = run_command("pstree -ap " + str(bashpid_tokill)
                                       + f" | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+' | "
                                         f"grep -oE '[0-9]+'  # RAM Monitor",
                                       debug_print=True)
            pids_to_monitor.remove(bashpid_tokill)
            if success:
                print(run_command_get_output(f'kill -s SIGINT {pid}  # RAM Monitor', True))
            else:
                print(f'DEBUG: RAM_USAGE: tmux session (with bash PID={bashpid_tokill}) already finished')
            time.sleep(2)
            break
        time.sleep(2)


def main():
    pass


if __name__ == '__main__':
    # Create the parser
    # REFER: https://realpython.com/command-line-interfaces-python-argparse/
    # REFER: https://stackoverflow.com/questions/19124304/what-does-metavar-and-action-mean-in-argparse-in-python
    # REFER: https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text
    # noinspection PyTypeChecker
    my_parser = argparse.ArgumentParser(
        prog='CalculateNetworkCost.py',
        description='Find cost of a network by executing various solvers',
        epilog=f"Note:"
               f"\n  • TODO: add message here"
               f"\n\nEnjoy the program :)",
        prefix_chars='-',
        fromfile_prefix_chars='@',
        allow_abbrev=False,
        add_help=True,
        formatter_class=argparse.RawTextHelpFormatter
    )
    my_parser.version = '1.0'

    # DIFFERENCE between Positional and Optional arguments
    #     Optional arguments start with - or --, while Positional arguments don't

    # Add the arguments
    my_parser.add_argument('--version', action='version')


    def parser_check_solver_models(val: str) -> str:
        # REFER: https://stackoverflow.com/questions/1265665/how-can-i-check-if-a-string-represents-an-int-without-using-try-except
        val_splitted = val.split()
        if len(val_splitted) == 0:
            raise argparse.ArgumentTypeError(f"no value passed")
        if len(val_splitted) == 1:
            if val_splitted[0] in AutoExecutorSettings.AVAILABLE_SOLVERS:
                raise argparse.ArgumentTypeError(f"no model numbers given")
            raise argparse.ArgumentTypeError(f"invalid solver name and no model number given")
        if val_splitted[0] not in AutoExecutorSettings.AVAILABLE_SOLVERS:
            raise argparse.ArgumentTypeError(f"invalid solver name")
        for i in val_splitted[1:]:
            if not i.isdigit():
                raise argparse.ArgumentTypeError(f"model number should be int")
            if int(i) not in AutoExecutorSettings.AVAILABLE_MODELS:
                raise argparse.ArgumentTypeError(f"invalid model number value: '{i}', "
                                                 f"valid values = {AutoExecutorSettings.AVAILABLE_MODELS}")
        return val


    my_parser.add_argument('--solver-models',
                           metavar='VAL',
                           action='append',
                           nargs='+',
                           type=parser_check_solver_models,
                           required=True,
                           help='Space separated `SOLVER_NAME MODEL_NUMBER [MODEL_NUMBER ...]`'
                                '\nNote:'
                                f'\n  • AVAILABLE SOLVERS = {AutoExecutorSettings.AVAILABLE_SOLVERS}'
                                f'\n  • AVAILABLE MODELS = {AutoExecutorSettings.AVAILABLE_MODELS}'
                                '\nExample Usage:\n  • --solver-models "baron 1 2 3 4" "octeract 1 2 3 4"')

    my_parser.add_argument('-j',
                           '--jobs',
                           metavar='N',
                           action='store',
                           # REFER: https://stackoverflow.com/questions/18700634/python-argparse-integer-condition-12
                           type=int,
                           default=0,
                           help='Max instances of solvers to execute in parallel [default: 0]'
                                '\nRequirement: N >= 0'
                                '\nNote:\n  • N=0 -> `nproc` or `len(os.sched_getaffinity(0))`')

    main()
