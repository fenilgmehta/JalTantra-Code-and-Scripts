#!/usr/bin/env python3
import argparse
import hashlib
import logging
import os
import subprocess
import sys
import time
from typing import List, Tuple, Union, Dict

from rich.logging import RichHandler as rich_RichHandler

g_logger = logging.getLogger('CNC')

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

def delete_last_lines(n=1):
    # REFER: https://www.quora.com/How-can-I-delete-the-last-printed-line-in-Python-language
    for _ in range(n):
        sys.stdout.write('\x1b[1A')  # Cursor up one line
        sys.stdout.write('\x1b[2K')  # Erase line


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


def file_md5(file_path) -> str:
    """It is assumed that the file will exist"""
    # REFER: https://stackoverflow.com/questions/16874598/how-do-i-calculate-the-md5-checksum-of-a-file-in-python
    global g_logger
    total_bytes = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        total_progress = min(len(chunk), total_bytes)
        last_progress_printed = -1
        while chunk:
            file_hash.update(chunk)
            new_progress = (100 * total_progress) // total_bytes
            if new_progress != last_progress_printed:
                if g_logger.getEffectiveLevel() <= logging.INFO:
                    delete_last_lines()
                    g_logger.info(f'Hash calculation {new_progress}% done')
                last_progress_printed = new_progress
            chunk = f.read(8192)
            total_progress = min(total_progress + len(chunk), total_bytes)
    return file_hash.hexdigest()


# ---

class AutoExecutorSettings:
    AVAILABLE_SOLVERS = ['baron', 'octeract']
    AVAILABLE_MODELS = {1: 'm1_basic.R', 2: 'm2_basic2_v2.R', 3: 'm3_descrete_segment.R', 4: 'm4_parallel_links.R'}

    def __init__(self):
        self.CPU_CORES_PER_SOLVER = 1
        # 48 core server is being used
        self.MAX_PARALLEL_SOLVERS = 44
        # Time is in seconds, set this to any value <= 0 to ignore this parameter
        self.EXECUTION_TIME_LIMIT = (0 * 60 * 60) + (5 * 60) + 0
        self.MIN_FREE_RAM = 2  # GiB
        self.MIN_FREE_SWAP = 8  # GiB, usefulness of this variable depends on the swappiness of the system

        self.output_dir = './NetworkResults/'.rstrip('/')  # Note: Do not put trailing forward slash ('/')
        self.output_data_dir = f'{self.output_dir}/SolutionData'
        self.models_dir = "./Files/Models"  # m1, m3 => q   ,   m2, m4 => q1, q2
        self.solvers = {}
        self.__update_solver_dict()

        # Tuples of (Solver name & Model name) which are to be executed to
        # find the cost of the given graph/network (i.e. data/testcase file)
        self.solver_model_combinations: List[Tuple[str, str]] = list()
        # Path to graph/network (i.e. data/testcase file)
        self.data_file_path: str = ''
        self.data_file_md5_hash: str = ''

    def set_execution_time_limit(self, hours: int = None, minutes: int = None, seconds: int = None) -> None:
        if (hours, minutes, seconds).count(None) == 3:
            print('At least one value should be non-None to update EXECUTION_TIME_LIMIT')
            return
        hours = 0 if hours is None else hours
        minutes = 0 if minutes is None else minutes
        seconds = 0 if seconds is None else seconds
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


def update_settings(args: argparse.Namespace):
    global g_logger, g_auto_executor_settings

    # noinspection PyArgumentList
    logging.basicConfig(
        level=(logging.DEBUG if args.debug else logging.WARNING),
        format='%(funcName)s :: %(message)s',
        datefmt="[%X]",
        handlers=[rich_RichHandler()]
    )
    g_logger = logging.getLogger('CNC')

    g_logger.debug(args)

    if not os.path.exists(args.path):
        print(f"Cannot access '{args.path}': No such file or directory")
        exit(1)
    g_auto_executor_settings.data_file_path = args.path
    g_auto_executor_settings.data_file_md5_hash = file_md5(args.path)
    g_logger.debug(f"Graph/Network (i.e. Data/Testcase file) = '{g_auto_executor_settings.data_file_path}'")
    g_logger.debug(f"Input file md5 = '{g_auto_executor_settings.data_file_md5_hash}'")

    g_auto_executor_settings.set_execution_time_limit(seconds=args.time)
    g_logger.debug(f'Execution Time = {g_auto_executor_settings.EXECUTION_TIME_LIMIT // 60 // 60:02}:'
                   f'{(g_auto_executor_settings.EXECUTION_TIME_LIMIT // 60) % 60:02}:'
                   f'{g_auto_executor_settings.EXECUTION_TIME_LIMIT % 60:02}')

    for solver_model_numbers_list in args.solver_models:
        for solver_model_numbers in solver_model_numbers_list:
            splitted_txt = solver_model_numbers.split()
            solver_name, model_numbers = splitted_txt[0], splitted_txt[1:]
            for i in model_numbers:
                g_auto_executor_settings.solver_model_combinations.append((
                    solver_name, AutoExecutorSettings.AVAILABLE_MODELS[int(i)]
                ))
    g_logger.debug(f'Solver Model Combinations = {g_auto_executor_settings.solver_model_combinations}')

    g_auto_executor_settings.CPU_CORES_PER_SOLVER = args.threads_per_solver_instance
    g_logger.debug(f'CPU_CORES_PER_SOLVER = {g_auto_executor_settings.CPU_CORES_PER_SOLVER}')

    if args.jobs == 0:
        g_auto_executor_settings.MAX_PARALLEL_SOLVERS = run_command_get_output('nproc')
    else:
        g_auto_executor_settings.MAX_PARALLEL_SOLVERS = args.jobs
    g_logger.debug(f'MAX_PARALLEL_SOLVERS = {g_auto_executor_settings.MAX_PARALLEL_SOLVERS}')


# REFER: https://stackoverflow.com/questions/1265665/how-can-i-check-if-a-string-represents-an-int-without-using-try-except
def parser_check_solver_models(val: str) -> str:
    val_splitted = val.split()
    if len(val_splitted) == 0:
        raise argparse.ArgumentTypeError(f"no value passed")
    if len(val_splitted) == 1:
        if val_splitted[0] in AutoExecutorSettings.AVAILABLE_SOLVERS:
            raise argparse.ArgumentTypeError(f"no model numbers given")
        raise argparse.ArgumentTypeError(f"invalid solver name")
    if val_splitted[0] not in AutoExecutorSettings.AVAILABLE_SOLVERS:
        raise argparse.ArgumentTypeError(f"invalid solver name")
    for i in val_splitted[1:]:
        if not i.isdigit():
            raise argparse.ArgumentTypeError(f"model number should be int")
        if int(i) not in AutoExecutorSettings.AVAILABLE_MODELS.keys():
            raise argparse.ArgumentTypeError(f"invalid model number value: '{i}', "
                                             f"valid values = {list(AutoExecutorSettings.AVAILABLE_MODELS.keys())}")
    return val


def parser_check_time_range(val: str) -> int:
    if val.count(':') != 2:
        raise argparse.ArgumentTypeError(f"invalid time value: '{val}', correct format is 'hh:mm:ss'")
    val_splitted = []
    # Handle inputs like '::30'
    for i in val.split(':'):
        val_splitted.append(i if len(i) > 0 else '0')
    for i in val_splitted:
        if not i.isdigit():
            raise argparse.ArgumentTypeError(f"invalid int value: '{val}'")
    if int(val_splitted[1]) >= 60:
        raise argparse.ArgumentTypeError(f"invalid minutes value: '{val}', 0 <= minutes < 60")
    if int(val_splitted[2]) >= 60:
        raise argparse.ArgumentTypeError(f"invalid seconds value: '{val}', 0 <= seconds < 60")
    seconds = int(val_splitted[0]) * 60 * 60 + int(val_splitted[1]) * 60 + int(val_splitted[2])
    if seconds < 30:
        raise argparse.ArgumentTypeError('minimum `N` is 30')
    return seconds


def parser_check_threads_int_range(c: str) -> int:
    if not c.isdigit():
        raise argparse.ArgumentTypeError(f"invalid int value: '{c}'")
    val = int(c)
    if val < 1:
        raise argparse.ArgumentTypeError('minimum `N` is 1')
    return val


def parser_check_jobs_int_range(c: str) -> int:
    if not c.isdigit():
        raise argparse.ArgumentTypeError(f"invalid int value: '{c}'")
    val = int(c)
    if val < 0:
        raise argparse.ArgumentTypeError('minimum `N` is 0')
    return val


if __name__ == '__main__':
    # Create the parser
    # REFER: https://realpython.com/command-line-interfaces-python-argparse/
    # REFER: https://stackoverflow.com/questions/19124304/what-does-metavar-and-action-mean-in-argparse-in-python
    # REFER: https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text
    # noinspection PyTypeChecker
    my_parser = argparse.ArgumentParser(
        prog='CalculateNetworkCost.py',
        description='Find cost of any graph/network (i.e. data/testcase file) '
                    'by executing various solvers using different models',
        epilog="Enjoy the program :)",
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

    my_parser.add_argument('-p',
                           '--path',
                           metavar='PATH',
                           action='store',
                           type=str,
                           required=True,
                           help='Path to graph/network (i.e. data/testcase file')

    my_parser.add_argument('--solver-models',
                           metavar='VAL',
                           action='append',
                           nargs='+',
                           type=parser_check_solver_models,
                           required=True,
                           help='Space separated `SOLVER_NAME MODEL_NUMBER [MODEL_NUMBER ...]`'
                                '\nNote:'
                                f'\n  • AVAILABLE SOLVERS = {AutoExecutorSettings.AVAILABLE_SOLVERS}'
                                f'\n  • AVAILABLE MODELS = {list(AutoExecutorSettings.AVAILABLE_MODELS.keys())}'
                                '\nExample Usage:\n  • --solver-models "baron 1 2 3 4" "octeract 1 2 3 4"')

    my_parser.add_argument('--time',
                           metavar='HH:MM:SS',
                           action='store',
                           # REFER: https://stackoverflow.com/questions/18700634/python-argparse-integer-condition-12
                           type=parser_check_time_range,
                           default=300,
                           help='Number of seconds a solver can execute [default: 00:05:00 = 5 min = 300 seconds]'
                                '\nRequirement: N >= 30 seconds')

    my_parser.add_argument('--threads-per-solver-instance',
                           metavar='N',
                           action='store',
                           # REFER: https://stackoverflow.com/questions/18700634/python-argparse-integer-condition-12
                           type=parser_check_threads_int_range,
                           default=1,
                           help='Set the number of threads a solver instance can have [default: 1]'
                                '\nRequirement: N >= 1')

    my_parser.add_argument('-j',
                           '--jobs',
                           metavar='N',
                           action='store',
                           # REFER: https://stackoverflow.com/questions/18700634/python-argparse-integer-condition-12
                           type=parser_check_jobs_int_range,
                           default=0,
                           help='Set maximum number of instances of solvers that can execute in parallel [default: 0]'
                                '\nRequirement: N >= 0'
                                '\nNote:\n  • N=0 -> `nproc` or `len(os.sched_getaffinity(0))`')

    my_parser.add_argument('--debug',
                           action='store_true',
                           help='Print debug information.')

    update_settings(my_parser.parse_args())
    main()
