#!/usr/bin/env python3
import argparse
import hashlib
import logging
import os
import pathlib
import subprocess
import sys
import time
from typing import List, Tuple, Union, Dict, Optional

from rich.logging import RichHandler as rich_RichHandler

g_logger = logging.getLogger('CNC')

# ---

# NOTE
#   1. • The prefix 'g_' denotes that it is a global variable.
#      • The prefix 'fn_' denotes that the variable stores a function.
#      • The prefix 'mas_' denotes that the function will do Monitoring and Stopping of running
#        solver instances depending on the conditions/parameters mentioned after this prefix.
#      • The prefix 'r_' denotes that the variable is for some system resource (CPU, RAM, time, ...)
#   2. 'pid_' and '.txt' are the prefix and suffix respectively
#      for text file having PID of the bash running inside tmux.
#   3. 'std_out_err_' and '.txt' are the prefix and suffix respectively for the text file
#      having the merged content of `stdout` and `stderr` stream of the tmux session which
#      runs the solver.
#   4. Tmux session prefix is 'AR_NC_'.
#   5. For naming files, always try to just use alpha-numeric letters and underscore ('_') only.


# Assumptions
#   1. Linux OS is used for execution
#   2. `bash`, `which`, `nproc`, `tmux` are installed
#   3. Python lib `rich` is installed
#   4. AMPL, Baron, Octeract are installed and properly configure
#      (Execution is done from "mtp" directory or any other directory with the same directory structure)
#   5. Model files are present at the right place
#   6. There is no limitation on the amount of available RAM (This assumption make the program simpler.
#      However, in future, it may be removed during deployment to make sure the solvers runs at optimal
#      speed - accordingly changes need to be done in this program)
#   7. Satisfy RegEx r'(a-zA-Z0-9_ )+' -> Absolute path of this Python script, and
#                                         absolute path to graph/network (i.e. data/testcase file)

# ---

# REFER: https://stackoverflow.com/questions/3172470/actual-meaning-of-shell-true-in-subprocess
g_BASH_PATH = subprocess.check_output(['which', 'bash'], shell=False).decode().strip()


def run_command(cmd: str, default_result: str = '', debug_print: bool = False) -> Tuple[bool, str]:
    """
    `stderr` is merged with `stdout`

    Returns:
        Tuple of [status, output]
    """
    # REFER: Context-Search-fms
    global g_logger, g_BASH_PATH
    if debug_print:
        g_logger.debug(f'COMMAND:\n`{cmd}`')
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
        if debug_print:
            g_logger.debug(f'OUTPUT:\n{output}')
        return True, output
    except subprocess.CalledProcessError as e:
        g_logger.info(f'EXCEPTION OCCURRED (cmd=`{cmd}`), will return '
                      f'default_result ("{default_result}") as the output')
        g_logger.info(f'CalledProcessError = {str(e).splitlines()[-1]}')
        g_logger.info(f'Exit Code = {e.returncode}')
        g_logger.info(f'Output = {e.output}')
        # g_logger.warning(e)
        # g_logger.warning(traceback.format_exc())
    if debug_print:
        g_logger.debug(default_result)
    return False, default_result


def run_command_get_output(cmd: str, default_result: str = '0', debug_print: bool = False) -> str:
    """
    The return value in case of unsuccessful execution of command `cmd` is '0', because sometimes we have used
    this method to get PID of some process and used kill command to send some signal (SIGINT in most cases) to
    that PID. If the command `cmd` which is used to find the PID of the target process fails, then in that case
    we return '0' so that kill command does not send the signal to any random process, instead it sends the
    signal to itself. Thus saving us from having to write `if` conditions which verify whether the PID is valid
    or not before executing the `kill` command.

    The `kill` commands differs with situation:
        1. kill --help
             (dev) ➜  ~ which kill
             kill: shell built-in command
        2. /bin/kill --help
             (dev) ➜  ~ env which kill      
             /bin/kill
        3. bash -c 'kill --help' (This has been used in this script)
             (dev) ➜  ~ bash -c 'which kill' 
             /bin/kill

    Returns:
        The return value is `default_result` (or '0') if the command `cmd` exits with a non-zero exit code.
        If command `cmd` executes successfully, then stdout and stderr are merged and returned as one string
    """
    return run_command(cmd, default_result, debug_print)[1]


# ---

def delete_last_lines(n=1):
    # REFER: https://www.quora.com/How-can-I-delete-the-last-printed-line-in-Python-language
    for _ in range(n):
        sys.stdout.write('\x1b[1A')  # Cursor up one line
        sys.stdout.write('\x1b[2K')  # Erase line


def get_free_ram() -> float:
    """Returns: free RAM in GiB"""
    # REFER: https://stackoverflow.com/questions/34937580/get-available-memory-in-gb-using-single-bash-shell-command/34938001
    return float(run_command_get_output(r'''awk '/MemFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo'''))


def get_free_swap() -> float:
    """Returns: free Swap in GiB"""
    # REFER: https://stackoverflow.com/questions/34937580/get-available-memory-in-gb-using-single-bash-shell-command/34938001
    return float(run_command_get_output(r'''awk '/SwapFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo'''))


def get_execution_time(pid: Union[int, str]) -> int:
    """Returns: execution time in seconds"""
    # REFER: https://unix.stackexchange.com/questions/7870/how-to-check-how-long-a-process-has-been-running
    return int(run_command(f'ps -o etimes= -p "{pid}"', str(10 ** 15))[1])  # 10**15 seconds == ~3.17 crore years


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

class SolverOutputAnalyzer:
    # Baron Functions below
    @staticmethod
    def baron_extract_output_table(std_out_err_file_path: str) -> str:
        return run_command_get_output(f"./output_table_extractor_baron.sh '{std_out_err_file_path}'", '')

    @staticmethod
    def baron_extract_best_solution(exec_info: 'NetworkExecutionInformation') -> Tuple[bool, float]:
        """The processing done by this function depends on the output format of `extract_solution_baron(...)` method"""
        global g_logger
        csv = SolverOutputAnalyzer.baron_extract_output_table(exec_info.uniq_std_out_err_file_path)
        if csv == '':
            return False, 0.0
        lines = csv.split('\n')
        lines = [line for line in lines if line != ',' and (not line.startswith('Processing file'))]
        g_logger.debug(f'{lines=}')

        status = len(lines) > 0
        best_solution = 0.0
        if len(lines) > 0:
            best_solution = float(lines[-1].split(',')[1])
        if best_solution > 1e40:
            g_logger.warning(f"Probably an infeasible solution found by Baron: '{lines[-1]}'")
            g_logger.info(f'Instance={exec_info.__str__()}')
            status = False
        return status, best_solution

    @staticmethod
    def baron_check_solution_found(exec_info: 'NetworkExecutionInformation') -> bool:
        return SolverOutputAnalyzer.baron_extract_best_solution(exec_info)[0]

    # Octeract Functions below
    @staticmethod
    def octeract_extract_output_table(std_out_err_file_path: str) -> str:
        return run_command_get_output(f"./output_table_extractor_octeract.sh '{std_out_err_file_path}'", '')

    @staticmethod
    def octeract_extract_best_solution(exec_info: 'NetworkExecutionInformation') -> Tuple[bool, float]:
        """The processing done by this function depends on the output format of `extract_solution_octeract(...)` method"""
        global g_logger
        csv = SolverOutputAnalyzer.octeract_extract_output_table(exec_info.uniq_std_out_err_file_path)
        if csv == '':
            return False, 0.0
        lines = csv.split('\n')
        lines = [line for line in lines if line != ',' and (not line.startswith('Processing file'))]
        g_logger.debug(f'{lines=}')

        status = len(lines) > 0
        best_solution = 0.0
        if len(lines) > 0:
            last_line_splitted = lines[-1].split(',')
            if len(last_line_splitted) > 2:
                g_logger.warning(f"Probably an infeasible solution found by Octeract: '{lines[-1]}'")
                g_logger.info(f'Instance={exec_info.__str__()}')
                status = False
            else:
                best_solution = float(last_line_splitted[1])
        return status, best_solution

    @staticmethod
    def octeract_check_solution_found(exec_info: 'NetworkExecutionInformation') -> bool:
        return SolverOutputAnalyzer.octeract_extract_best_solution(exec_info)[0]


# ---

class NetworkExecutionInformation:
    def __init__(self, aes: 'AutoExecutorSettings', idx: int):
        """
        This constructor will set all data members except `self.tmux_bash_pid`
        """
        global g_logger

        self.tmux_bash_pid: Union[str, None] = None  # This has to be set manually
        self.idx: int = idx
        self.solver_name, self.model_name = aes.solver_model_combinations[idx]
        self.solver_info: SolverInformation = aes.solvers[self.solver_name]

        # REFER: https://stackoverflow.com/a/66771847
        self.short_uniq_model_name: str = self.model_name[:self.model_name.find('_')]
        # REFER: https://github.com/tmux/tmux/issues/3113
        #        Q. What is the maximum length of session name that can be set using
        #           the following command: `tmux new -s 'SessionName_12345'`
        #        A. There is no defined limit.
        self.short_uniq_data_file_name: str = aes.data_file_md5_hash
        self.short_uniq_combination: str = f'{self.solver_name}_' \
                                           f'{self.short_uniq_model_name}_{self.short_uniq_data_file_name}'
        g_logger.debug(self.short_uniq_model_name, self.short_uniq_data_file_name, self.short_uniq_combination)

        self.models_dir: str = aes.models_dir
        self.data_file_path: str = aes.data_file_path
        self.engine_path: str = aes.solvers[self.solver_name].engine_path
        self.engine_options: str = aes.solvers[self.solver_name].engine_options
        self.uniq_exec_output_dir: pathlib.Path = \
            pathlib.Path(aes.output_dir_level_1_network_specific) / self.short_uniq_combination

        self.uniq_tmux_session_name: str = f'{aes.TMUX_UNIQUE_PREFIX}{self.short_uniq_combination}'
        self.uniq_pid_file_path: str = f'/tmp/pid_{self.short_uniq_combination}.txt'
        self.uniq_std_out_err_file_path: str = f'{self.uniq_exec_output_dir.resolve()}/std_out_err.txt'

    def __str__(self):
        return f'NetworkExecutionInformation[pid={self.tmux_bash_pid}, idx={self.idx}, solver={self.solver_name}, ' \
               f'model={self.short_uniq_model_name}]'


class SolverInformation:
    def __init__(self, engine_path: str, engine_options: str, process_name_to_stop_using_ctrl_c: str,
                 fn_check_solution_found, fn_extract_best_solution):
        """
        Args:
            engine_path: Path to the solver that will be used by AMPL
            engine_options: Solver specific parameters in AMPL format
            process_name_to_stop_using_ctrl_c: Name of the process that is to be stopped using
                                               Ctrl+C (i.e. SIGINT signal) such that solver smartly
                                               gives us the best solution found till that moment
            fn_check_solution_found: This should be a function that accepts a variable
                                     of type `class NetworkExecutionInformation`
        """
        self.engine_path = engine_path
        self.engine_options = engine_options
        self.process_name_to_stop_using_ctrl_c = process_name_to_stop_using_ctrl_c
        self.fn_check_solution_found = fn_check_solution_found
        self.fn_extract_best_solution = fn_extract_best_solution

    def check_solution_found(self, exec_info: NetworkExecutionInformation) -> bool:
        """
        Parses the output (stdout and stderr) of the solver and tells us
        whether the solver has found any feasible solution or not

        Args:
            exec_info: NetworkExecutionInformation object having all information regarding the execution of the solver

        Returns:
             A boolean value telling whether the solver found any feasible solution or not
        """
        global g_logger
        if self.fn_check_solution_found is None:
            g_logger.error(f"`self.fn_check_solution_found` is `None` for self.engine_path='{self.engine_path}'")
            return True
        return self.fn_check_solution_found(exec_info)

    def extract_best_solution(self, exec_info: NetworkExecutionInformation) -> Tuple[bool, float]:
        """
        Parses the output (stdout and stderr) of the solver and tells us
        whether the solver has found any feasible solution or not, and if
        it has, then return its value as well.

        Args:
            exec_info: NetworkExecutionInformation object having all information regarding the execution of the solver

        Returns:
             A boolean value telling whether the solver found any feasible solution or not
             A float value which is the optimal solution found till that moment
        """
        global g_logger
        if self.fn_extract_best_solution is None:
            g_logger.error(f"`self.fn_check_solution_found` is `None` for self.engine_path='{self.engine_path}'")
            return True, 0.0
        return self.fn_extract_best_solution(exec_info)


class AutoExecutorSettings:
    # Level 0 is main directory inside which everything will exist
    OUTPUT_DIR_LEVEL_0 = './NetworkResults/'.rstrip('/')  # Note: Do not put trailing forward slash ('/')
    OUTPUT_DIR_LEVEL_1_DATA = f'{OUTPUT_DIR_LEVEL_0}/SolutionData'
    # Please ensure that proper escaping of white spaces and other special characters
    # is done because this will be executed in a fashion similar to `./a.out`
    AMPL_PATH = './ampl.linux-intel64/ampl'
    AVAILABLE_SOLVERS = ['baron', 'octeract']
    AVAILABLE_MODELS = {1: 'm1_basic.R', 2: 'm2_basic2_v2.R', 3: 'm3_descrete_segment.R', 4: 'm4_parallel_links.R'}
    TMUX_UNIQUE_PREFIX = f'AR_NC_{os.getpid()}_'  # AR = Auto Run, NC = Network Cost

    def __init__(self):
        self.r_cpu_cores_per_solver = 1
        # 48 core server is being used
        self.r_max_parallel_solvers = 44
        # Time is in seconds, set this to any value <= 0 to ignore this parameter
        self.r_execution_time_limit = (0 * 60 * 60) + (5 * 60) + 0
        self.r_min_free_ram = 2  # GiB
        self.r_min_free_swap = 8  # GiB, usefulness of this variable depends on the swappiness of the system

        self.models_dir = "./Files/Models"  # m1, m3 => q   ,   m2, m4 => q1, q2
        self.solvers: Dict[str, SolverInformation] = {}
        self.__update_solver_dict()

        # Tuples of (Solver name & Model name) which are to be executed to
        # find the cost of the given graph/network (i.e. data/testcase file)
        self.solver_model_combinations: List[Tuple[str, str]] = list()
        # Path to graph/network (i.e. data/testcase file)
        self.data_file_path: str = ''
        self.data_file_md5_hash: str = ''
        self.output_dir_level_1_network_specific: str = ''
        self.output_network_specific_result: str = ''

    def __update_solver_dict(self):
        # NOTE: Update `AutoExecutorSettings.AVAILABLE_SOLVERS` if keys in below dictionary are updated
        # NOTE: Use double quotes ONLY in the below variables
        self.solvers = {
            'baron': SolverInformation(
                engine_path='./ampl.linux-intel64/baron',
                engine_options=f'option baron_options "maxtime={self.r_execution_time_limit - 10} '
                               f'threads={self.r_cpu_cores_per_solver} barstats keepsol lsolmsg '
                               f'outlev=1 prfreq=100 prtime=2 problem";',
                process_name_to_stop_using_ctrl_c='baron',  # For 1 core and multi core, same process is to be stopped
                fn_check_solution_found=SolverOutputAnalyzer.baron_check_solution_found,
                fn_extract_best_solution=SolverOutputAnalyzer.baron_extract_best_solution
            ),
            'octeract': SolverInformation(
                engine_path='./octeract-engine-4.0.0/bin/octeract-engine',
                engine_options=f'options octeract_options "num_cores={self.r_cpu_cores_per_solver}";',
                # For 1 core, process with name 'octeract-engine' is the be stopped using Control+C
                # For multi core, process with name 'mpirun' is the be stopped using Control+C
                process_name_to_stop_using_ctrl_c='mpirun' if self.r_cpu_cores_per_solver > 1 else 'octeract-engine',
                fn_check_solution_found=SolverOutputAnalyzer.octeract_check_solution_found,
                fn_extract_best_solution=SolverOutputAnalyzer.octeract_extract_best_solution
            )
        }

    def set_execution_time_limit(self, hours: int = None, minutes: int = None, seconds: int = None) -> None:
        if (hours, minutes, seconds).count(None) == 3:
            g_logger.warning('At least one value should be non-None to update EXECUTION_TIME_LIMIT')
            return
        hours = 0 if hours is None else hours
        minutes = 0 if minutes is None else minutes
        seconds = 0 if seconds is None else seconds
        self.r_execution_time_limit = (hours * 60 * 60) + (minutes * 60) + seconds
        self.__update_solver_dict()

    def set_cpu_cores_per_solver(self, n: int) -> None:
        self.r_cpu_cores_per_solver = n
        self.__update_solver_dict()

    def set_data_file_path(self, data_file_path: str) -> None:
        self.data_file_path = data_file_path
        self.data_file_md5_hash = file_md5(data_file_path)
        self.output_dir_level_1_network_specific = f'{AutoExecutorSettings.OUTPUT_DIR_LEVEL_0}' \
                                                   f'/{self.data_file_md5_hash}'
        self.output_network_specific_result = self.output_dir_level_1_network_specific + '/0-result.txt'

    def start_solver(self, idx: int) -> NetworkExecutionInformation:
        """
        Launch the solver using `tmux` and `AMPL` in background (i.e. asynchronously / non-blocking)

        Args:
            idx: Index of `self.solver_model_combinations`

        Returns:
            `class NetworkExecutionInformation` object which has all the information regarding the execution
        """
        info = NetworkExecutionInformation(self, idx)

        info.uniq_exec_output_dir.mkdir(exist_ok=True)
        if not info.uniq_exec_output_dir.exists():
            g_logger.warning(f"Some directory(s) do not exist in the path: '{info.uniq_exec_output_dir.resolve()}'")
            info.uniq_exec_output_dir.mkdir(parents=True, exist_ok=True)

        # REFER: https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash
        #        📝 'EOF' should be the only word on the line without any space before and after it.
        # NOTE: The statement `echo > /dev/null` is required to make the below command work. Without
        #       it, AMPL is not started. Probably, it has something to do with the `EOF` thing.
        # NOTE: The order of > and 2>&1 matters in the below command
        run_command_get_output(rf'''
            tmux new-session -d -s '{info.uniq_tmux_session_name}' '
echo $$ > "{info.uniq_pid_file_path}"
{self.AMPL_PATH} > "{info.uniq_std_out_err_file_path}" 2>&1 <<EOF
    reset;
    model "{info.models_dir}/{info.model_name}";
    data "{info.data_file_path}";
    option solver "{info.engine_path}";
    {info.engine_options};
    solve;
    display _total_solve_time;
    display l;
    display {"q1,q2" if (info.short_uniq_model_name in ("m2", "m4")) else "q"};
EOF
echo > /dev/null
'
        ''', debug_print=self.debug)

        info.tmux_bash_pid = run_command_get_output(f'cat "/tmp/pid_{info.short_uniq_combination}.txt"')
        return info


g_settings = AutoExecutorSettings()


# ---

class MonitorAndStopper:
    @staticmethod
    def mas_time(
            tmux_monitor_list: List[NetworkExecutionInformation],
            tmux_finished_list: List[NetworkExecutionInformation],
            execution_time_limit: float,
            blocking: bool
    ) -> None:
        """
        Monitor and stop solver instances based on the time for which they have been running on the system

        Args:
            tmux_monitor_list: List of Solver instances which are to be monitored
            tmux_finished_list: List of Solver instances which have been stopped in this iteration (initially this will be empty)
            execution_time_limit: Time in seconds. (This should be > 0)
            blocking: Wait until one of the solver instance in `tmux_monitor_list` is stopped
        """
        if execution_time_limit <= 0.0:
            g_logger.error(f'FIXME: `execution_time_limit` is not greater than 0')
            return
        # NOTE: It is the below TODO item which resulted in lots of changes
        #       to the code and in turn improving the program structure :)
        # TODO: Dynamically find the value of `process_name_to_stop_using_ctrl_c` based on solver name and PID
        #       May have to update the `pids_to_monitor` list to store solver name along with the PID

        # Index of elements of `tmux_monitor_list` which were/have stopped.
        tmux_finished_list_idx: List[int] = list()
        to_run_the_loop = True
        while to_run_the_loop:
            to_run_the_loop = blocking
            for i, ne_info in enumerate(tmux_monitor_list):
                if get_execution_time(ne_info.tmux_bash_pid) < execution_time_limit:
                    continue
                # NOTE: only SIGINT signal (i.e. Ctrl+C) does proper termination of the octeract-engine
                g_logger.debug(run_command_get_output(
                    f"pstree -ap {ne_info.tmux_bash_pid}  # Time 1", debug_print=True
                ))
                g_logger.debug(run_command_get_output(
                    f"pstree -ap {ne_info.tmux_bash_pid} | "
                    f"grep -oE '{ne_info.solver_info.process_name_to_stop_using_ctrl_c},[0-9]+'  # Time 2"
                ))
                g_logger.debug(run_command_get_output(
                    f"pstree -aps {ne_info.tmux_bash_pid} | "
                    f"grep -oE '{ne_info.solver_info.process_name_to_stop_using_ctrl_c},[0-9]+'  # Time 3"
                ))
                success, pid = run_command(
                    f"pstree -ap {ne_info.tmux_bash_pid} | "
                    f"grep -oE '{ne_info.solver_info.process_name_to_stop_using_ctrl_c},[0-9]+' | "
                    f"grep -oE '[0-9]+'  # Time Monitor 4",
                    '0',
                    True
                )
                tmux_finished_list_idx.append(i)
                tmux_finished_list.append(ne_info)
                to_run_the_loop = False
                if success:
                    g_logger.info(run_command_get_output(f'kill -s SIGINT {pid}  # Time Monitor', debug_print=True))
                else:
                    g_logger.info(f'TIME_LIMIT: tmux session (with bash PID={ne_info.tmux_bash_pid}) already finished')
                time.sleep(2)
            for i in tmux_finished_list_idx[::-1]:
                tmux_monitor_list.pop(i)
            time.sleep(2)
        g_logger.debug(f'{tmux_finished_list_idx=}')
        pass


def check_solution_status(tmux_monitor_list: List[NetworkExecutionInformation]) -> bool:
    global g_settings
    # https://stackoverflow.com/questions/878943/why-return-notimplemented-instead-of-raising-notimplementederror
    for info in tmux_monitor_list:
        if g_settings.solvers[info.solver_name].check_solution_found(info):
            return True
    return False


def extract_best_solution(tmux_monitor_list: List[NetworkExecutionInformation]) -> \
        Tuple[bool, float, Optional[NetworkExecutionInformation]]:
    """
    Args:
        tmux_monitor_list: List of `NetworkExecutionInformation` which have finished their execution

    Returns:
        status, best solution, context of solver and model which found the best solution
    """
    global g_settings
    best_result_till_now, best_result_exec_info = float('inf'), None
    for exec_info in tmux_monitor_list:
        status, curr_res = g_settings.solvers[exec_info.solver_name].extract_best_solution(exec_info)
        g_logger.debug(f'solver={exec_info.solver_name}, model={exec_info.short_uniq_model_name}, {status=}')
        # `if` solution not found by this solver instance `or` a better solution is already known, then `continue`
        if not status or curr_res > best_result_till_now:
            continue
        g_logger.debug(f'Update best result seen till now: {curr_res} <= {best_result_till_now=}')
        best_result_till_now = curr_res
        best_result_exec_info = exec_info
    return best_result_exec_info is not None, best_result_till_now, best_result_exec_info


def main():
    global g_logger, g_settings
    run_command_get_output(f'mkdir -p "{g_settings.OUTPUT_DIR_LEVEL_0}"')
    run_command_get_output(f'mkdir -p "{g_settings.OUTPUT_DIR_LEVEL_1_DATA}"')
    run_command_get_output(f'mkdir -p "{g_settings.output_dir_level_1_network_specific}"')

    tmux_monitor_list: List[NetworkExecutionInformation] = list()
    tmux_finished_list: List[NetworkExecutionInformation] = list()

    min_combination_parallel_solvers = min(
        len(g_settings.solver_model_combinations),
        g_settings.r_max_parallel_solvers
    )

    for i in range(min_combination_parallel_solvers):
        exec_info = g_settings.start_solver(i)
        tmux_monitor_list.append(exec_info)
        g_logger.info(f'tmux session "{exec_info.short_uniq_combination}" -> {exec_info.tmux_bash_pid}')
        time.sleep(0.2)

    # TODO: Problem: Handle case of deadlock like situation
    #         1. `g_settings.solver_model_combinations > g_settings.MAX_PARALLEL_SOLVERS`
    #         2. The first `g_settings.MAX_PARALLEL_SOLVERS` solver model combinations are poor and
    #            the solver is unable to find any feasible solution even after executing for hours
    #       Solution 1: Add a flag to impose hard deadline on execution time, i.e. the execution of a solver is
    #                   to be stopped if no solution is found by it within the specified hard deadline timelimit
    # TODO: Problem: See if we can optimise the execution in the below situation:
    #         1. `g_settings.solver_model_combinations > g_settings.MAX_PARALLEL_SOLVERS`
    #         2. Some error occur in the execution of a one of
    #            `g_settings.solver_model_combinations[:g_settings.MAX_PARALLEL_SOLVERS]`
    #            way before `g_settings.EXECUTION_TIME_LIMIT`
    time.sleep(g_settings.r_execution_time_limit)
    at_least_one_solution_found = False
    while not check_solution_status(tmux_monitor_list):
        time.sleep(10)  # Give 10 more seconds to the running solvers
    at_least_one_solution_found = True

    for i in range(min_combination_parallel_solvers, len(g_settings.solver_model_combinations)):
        g_logger.debug(run_command_get_output(f'tmux ls | grep "{g_settings.TMUX_UNIQUE_PREFIX}"'))
        tmux_sessions_running = int(run_command_get_output(f'tmux ls | grep "{g_settings.TMUX_UNIQUE_PREFIX}" | wc -l'))
        g_logger.debug(tmux_sessions_running)

        while tmux_sessions_running >= g_settings.r_max_parallel_solvers:
            g_logger.debug("----------")
            g_logger.debug(f'{tmux_monitor_list=}')
            g_logger.debug(f'{len(tmux_finished_list)=}')
            MonitorAndStopper.mas_time(tmux_monitor_list, tmux_finished_list, g_settings.r_execution_time_limit, True)
            g_logger.debug(f'{tmux_monitor_list=}')
            g_logger.debug(f'{len(tmux_finished_list)=}')

        exec_info = g_settings.start_solver(i)
        tmux_monitor_list.append(exec_info)
        g_logger.info(f'tmux session "{exec_info.short_uniq_combination}" -> {exec_info.tmux_bash_pid}')
        time.sleep(0.2)

    while int(run_command_get_output(f'tmux ls | grep "{g_settings.TMUX_UNIQUE_PREFIX}" | wc -l')) \
            >= g_settings.r_max_parallel_solvers:
        g_logger.debug("----------")
        g_logger.debug(f'{tmux_monitor_list=}')
        g_logger.debug(f'{len(tmux_finished_list)=}')
        MonitorAndStopper.mas_time(tmux_monitor_list, tmux_finished_list, g_settings.r_execution_time_limit, True)
        g_logger.debug(f'{tmux_monitor_list=}')
        g_logger.debug(f'{len(tmux_finished_list)=}')

    run_command_get_output(f"cp -r /tmp/at*nl /tmp/at*octsol /tmp/baron_tmp* '{g_settings.OUTPUT_DIR_LEVEL_1_DATA}'")

    # TODO: Get the best result and its solution file
    status, best_cost, best_cost_instance_exec_info = extract_best_solution(tmux_finished_list)
    if not status:
        g_logger.error('NO feasible solution found')
        run_command(f"touch '{g_settings.output_network_specific_result}'")
    else:
        g_logger.info(f'{best_cost=}')
        g_logger.info(f'Instance={best_cost_instance_exec_info.__str__()}')
        g_logger.info(f'Solver={best_cost_instance_exec_info.solver_name}, '
                      f'Model={best_cost_instance_exec_info.short_uniq_model_name}')
        run_command(f"echo '{best_cost}' >> '{g_settings.output_network_specific_result}'")
        run_command(f"echo '{best_cost_instance_exec_info.solver_name}'"
                    f" >> '{g_settings.output_network_specific_result}'")
        run_command(f"echo '{best_cost_instance_exec_info.short_uniq_model_name}'"
                    f" >> '{g_settings.output_network_specific_result}'")
        run_command(f"echo '{best_cost_instance_exec_info.uniq_std_out_err_file_path}'"
                    f" >> '{g_settings.output_network_specific_result}'")
    pass


def update_settings(args: argparse.Namespace):
    global g_logger, g_settings

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
        g_logger.error(f"Cannot access '{args.path}': No such file or directory")
        exit(1)
    g_settings.set_data_file_path(args.path)
    g_logger.debug(f"Graph/Network (i.e. Data/Testcase file) = '{g_settings.data_file_path}'")
    g_logger.debug(f"Input file md5 = '{g_settings.data_file_md5_hash}'")

    g_settings.set_execution_time_limit(seconds=args.time)
    g_logger.debug(f'Solver Execution Time Limit = {g_settings.r_execution_time_limit // 60 // 60:02}:'
                   f'{(g_settings.r_execution_time_limit // 60) % 60:02}:'
                   f'{g_settings.r_execution_time_limit % 60:02}')

    for solver_model_numbers_list in args.solver_models:
        for solver_model_numbers in solver_model_numbers_list:
            splitted_txt = solver_model_numbers.split()
            solver_name, model_numbers = splitted_txt[0], splitted_txt[1:]
            for i in model_numbers:
                g_settings.solver_model_combinations.append((
                    solver_name, AutoExecutorSettings.AVAILABLE_MODELS[int(i)]
                ))
    g_logger.debug(f'Solver Model Combinations = {g_settings.solver_model_combinations}')

    g_settings.set_cpu_cores_per_solver(args.threads_per_solver_instance)
    g_logger.debug(f'r_cpu_cores_per_solver = {g_settings.r_cpu_cores_per_solver}')

    if args.jobs == 0:
        g_settings.r_max_parallel_solvers = len(g_settings.solver_model_combinations)
    elif args.jobs == -1:
        g_settings.r_max_parallel_solvers = run_command_get_output('nproc')
    else:
        g_settings.r_max_parallel_solvers = args.jobs
    g_logger.debug(f'r_max_parallel_solvers = {g_settings.r_max_parallel_solvers}')
    if g_settings.r_max_parallel_solvers < len(g_settings.solver_model_combinations):
        # TODO: Add more clear warning message explaining the technique used to get the results
        #       Result = Return the best result found in `EXECUTION_TIME_LIMIT` time among all solver model combinations.
        #                If no result is found, then wait until the first result is found and then return it.
        g_logger.warning('There is a possibility of more time being spent on execution'
                         'as all solver model combinations will not be running in parallel.'
                         f'\nSolver Model Combinations = {len(g_settings.solver_model_combinations)}')


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
    if val < -1:
        raise argparse.ArgumentTypeError('minimum `N` is -1')
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
                                '\nRequirement: N >= -1'
                                '\nNote:'
                                '\n  • N=0 -> Number of solver model combinations due to `--solver-models` parameter'
                                '\n  • N=-1 -> `nproc` or `len(os.sched_getaffinity(0))`')

    my_parser.add_argument('--debug',
                           action='store_true',
                           help='Print debug information.')

    update_settings(my_parser.parse_args())
    main()
