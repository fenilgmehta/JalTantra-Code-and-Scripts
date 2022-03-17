#!/usr/bin/env python3
import argparse
import hashlib
import logging
import os
import pathlib
import subprocess
import sys
import time
from typing import List, Tuple, Union, Dict

from rich.logging import RichHandler as rich_RichHandler

g_logger = logging.getLogger('CNC')

# ---

# NOTE
#   1. • The prefix 'g_' denotes that it is a global variable.
#      • The prefix 'fn_' denotes that the variable stores a function.
#      • The prefix 'mas_' denotes that the function will do Monitoring and Stopping of running
#        solver instances depending on the conditions/parameters mentioned after this prefix.
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
        g_logger.debug(f'COMMAND: `{cmd}`')
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
            g_logger.debug(output)
        return True, output
    except Exception as e:
        g_logger.warning(f'EXCEPTION OCCURRED (cmd=`{cmd}`), will return '
                         f'default_result ("{default_result}") as the output')
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
        return len(lines) > 0, lines[-1] if len(lines) > 0 else 0.0

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
        return len(lines) > 0, lines[-1] if len(lines) > 0 else 0.0

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
        self.output_dir: pathlib.Path = pathlib.Path(aes.output_dir) / self.short_uniq_combination

        self.uniq_tmux_session_name: str = f'{aes.TMUX_UNIQUE_PREFIX}{self.short_uniq_combination}'
        self.uniq_pid_file_path: str = f'/tmp/pid_{self.short_uniq_combination}.txt'
        self.uniq_output_file_path: str = f'{self.output_dir.resolve()}/std_out_err_{self.short_uniq_combination}.txt'

    def __str__(self):
        return f'[pid={self.tmux_bash_pid}, idx={self.idx}, solver={self.solver_name}, ' \
               f'model={self.short_uniq_model_name}]'


class SolverInformation:
    def __init__(self, engine_path: str, engine_options: str, process_name_to_stop_using_ctrl_c: str,
                 fn_check_solution_found):
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

    def check_solution_found(self, exec_info: NetworkExecutionInformation) -> bool:
        """
        Parses the output (stdout and stderr) of the solver and tells us
        whether the solver has found any feasible solution or not

        Args:
            exec_info: NetworkExecutionInformation object having all information regarding the execution of the solver

        Returns:
             A boolean value telling whether the solver found any feasible solution or not
        """
        # REFER: https://stackoverflow.com/questions/42499656/pass-all-arguments-of-a-function-to-another-function
        global g_logger
        if self.fn_check_solution_found is None:
            g_logger.error(f"`self.fn_check_solution_found` is `None` for self.engine_path='{self.engine_path}'")
            return True
        return self.fn_check_solution_found(exec_info)


class AutoExecutorSettings:
    # Please ensure that proper escaping of white spaces and other special characters
    # is done because this will be executed in a fashion similar to `./a.out`
    AMPL_PATH = './ampl.linux-intel64/ampl'
    AVAILABLE_SOLVERS = ['baron', 'octeract']
    AVAILABLE_MODELS = {1: 'm1_basic.R', 2: 'm2_basic2_v2.R', 3: 'm3_descrete_segment.R', 4: 'm4_parallel_links.R'}
    TMUX_UNIQUE_PREFIX = f'AR_NC_{os.getpid()}_'  # AR = Auto Run, NC = Network Cost

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
        self.solvers: Dict[str, SolverInformation] = {}
        self.__update_solver_dict()

        # Tuples of (Solver name & Model name) which are to be executed to
        # find the cost of the given graph/network (i.e. data/testcase file)
        self.solver_model_combinations: List[Tuple[str, str]] = list()
        # Path to graph/network (i.e. data/testcase file)
        self.data_file_path: str = ''
        self.data_file_md5_hash: str = ''

    def __update_solver_dict(self):
        # NOTE: Update `AutoExecutorSettings.AVAILABLE_SOLVERS` if keys in below dictionary are updated
        # NOTE: Use double quotes ONLY in the below variables
        self.solvers = {
            'baron': SolverInformation(
                engine_path='./ampl.linux-intel64/baron',
                engine_options=f'option baron_options "maxtime={self.EXECUTION_TIME_LIMIT - 10} '
                               f'threads={self.CPU_CORES_PER_SOLVER} barstats keepsol lsolmsg '
                               f'outlev=1 prfreq=100 prtime=2 problem";',
                process_name_to_stop_using_ctrl_c='baron',  # For 1 core and multi core, same process is to be stopped
                fn_check_solution_found=None
            ),
            'octeract': SolverInformation(
                engine_path='./octeract-engine-4.0.0/bin/octeract-engine',
                engine_options=f'options octeract_options "num_cores={self.CPU_CORES_PER_SOLVER}";',
                # For 1 core, process with name 'octeract-engine' is the be stopped using Control+C
                # For multi core, process with name 'mpirun' is the be stopped using Control+C
                process_name_to_stop_using_ctrl_c='mpirun' if self.CPU_CORES_PER_SOLVER > 1 else 'octeract-engine',
                fn_check_solution_found=None
            )
        }

    def set_execution_time_limit(self, hours: int = None, minutes: int = None, seconds: int = None) -> None:
        if (hours, minutes, seconds).count(None) == 3:
            g_logger.warning('At least one value should be non-None to update EXECUTION_TIME_LIMIT')
            return
        hours = 0 if hours is None else hours
        minutes = 0 if minutes is None else minutes
        seconds = 0 if seconds is None else seconds
        self.EXECUTION_TIME_LIMIT = (hours * 60 * 60) + (minutes * 60) + seconds
        self.__update_solver_dict()

    def set_cpu_cores_per_solver(self, n: int) -> None:
        self.CPU_CORES_PER_SOLVER = n
        self.__update_solver_dict()

    def start_solver(self, idx: int) -> NetworkExecutionInformation:
        """
        Launch the solver using `tmux` and `AMPL` in background (i.e. asynchronously / non-blocking)

        Args:
            idx: Index of `self.solver_model_combinations`

        Returns:
            `class NetworkExecutionInformation` object which has all the information regarding the execution
        """
        info = NetworkExecutionInformation(self, idx)

        info.output_dir.mkdir(exist_ok=True)
        if not info.output_dir.exists():
            g_logger.warning(f"Some directory(s) do not exist in the path: '{info.output_dir.resolve()}'")
            info.output_dir.mkdir(parents=True, exist_ok=True)

        # NOTE: The order of > and 2>&1 matters in the below command
        run_command_get_output(rf'''
            tmux new-session -d -s '{info.uniq_tmux_session_name}' 'echo $$ > '{info.uniq_pid_file_path}' ; {self.AMPL_PATH} > '{info.uniq_output_file_path}' 2>&1 <<EOF
                reset;
                model "{info.models_dir}/{info.model_name}";
                data "{info.data_file_path}";
                option solver "{info.engine_path}";
                {info.engine_options};
                solve;
                display _total_solve_time;
                display l;
                display {"q1,q2" if (info.short_uniq_model_name in ("m2", "m4")) else "q"};
            EOF'
        ''')

        info.tmux_bash_pid = run_command_get_output(f'cat "/tmp/pid_{info.short_uniq_combination}.txt"')
        return info


g_settings = AutoExecutorSettings()


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
                    # NOTE: only SIGINT signal (i.e. Ctrl+C) does proper termination of the octeract-engine
                    g_logger.debug(run_command_get_output(f"pstree -ap {i_bashpid}  # Time 1", debug_print=True))
                    g_logger.debug(run_command_get_output(
                        f"pstree -ap {i_bashpid} | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # Time 2"
                    ))
                    g_logger.debug(run_command_get_output(
                        f"pstree -aps {i_bashpid} | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # Time 3"
                    ))
                    success, pid = run_command(f"pstree -ap {i_bashpid} | "
                                               f"grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+' | "
                                               f"grep -oE '[0-9]+'  # Time Monitor 4",
                                               '0',
                                               True)
                    pids_finished.append(i_bashpid)
                    to_run_the_loop = False
                    if success:
                        g_logger.info(run_command_get_output(f'kill -s SIGINT {pid}  # Time Monitor', debug_print=True))
                    else:
                        g_logger.info(f'TIME_LIMIT: tmux session (with bash PID={i_bashpid}) already finished')
                    time.sleep(2)
            for i_bashpid in pids_finished:
                pids_to_monitor.remove(i_bashpid)
            pids_finished.clear()
        if get_free_ram() <= min_free_ram:
            # Kill the oldest executing octeract instance used to solve data+model combination
            bashpid_tokill = sorted([(get_execution_time(p), p) for p in pids_to_monitor], reverse=True)[0][1]
            g_logger.debug(run_command_get_output(
                f"pstree -ap {bashpid_tokill} | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # RAM 1"
            ))
            g_logger.debug(run_command_get_output(
                f"pstree -aps {bashpid_tokill} | grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+'  # RAM 2"
            ))
            success, pid = run_command(f"pstree -ap {bashpid_tokill} | "
                                       f"grep -oE '{process_name_to_stop_using_ctrl_c},[0-9]+' | "
                                       f"grep -oE '[0-9]+'  # RAM Monitor 3",
                                       '0',
                                       True)
            pids_to_monitor.remove(bashpid_tokill)
            if success:
                g_logger.info(run_command_get_output(f'kill -s SIGINT {pid}  # RAM Monitor', debug_print=True))
            else:
                g_logger.info(f'RAM_USAGE: tmux session (with bash PID={bashpid_tokill}) already finished')
            time.sleep(2)
            break
        time.sleep(2)


def main():
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
    g_settings.data_file_path = args.path
    g_settings.data_file_md5_hash = file_md5(args.path)
    g_logger.debug(f"Graph/Network (i.e. Data/Testcase file) = '{g_settings.data_file_path}'")
    g_logger.debug(f"Input file md5 = '{g_settings.data_file_md5_hash}'")

    g_settings.set_execution_time_limit(seconds=args.time)
    g_logger.debug(f'Solver Execution Timelimit = {g_settings.EXECUTION_TIME_LIMIT // 60 // 60:02}:'
                   f'{(g_settings.EXECUTION_TIME_LIMIT // 60) % 60:02}:'
                   f'{g_settings.EXECUTION_TIME_LIMIT % 60:02}')

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
    g_logger.debug(f'CPU_CORES_PER_SOLVER = {g_settings.CPU_CORES_PER_SOLVER}')

    if args.jobs == 0:
        g_settings.MAX_PARALLEL_SOLVERS = len(g_settings.solver_model_combinations)
    elif args.jobs == -1:
        g_settings.MAX_PARALLEL_SOLVERS = run_command_get_output('nproc')
    else:
        g_settings.MAX_PARALLEL_SOLVERS = args.jobs
    g_logger.debug(f'MAX_PARALLEL_SOLVERS = {g_settings.MAX_PARALLEL_SOLVERS}')
    if g_settings.MAX_PARALLEL_SOLVERS < len(g_settings.solver_model_combinations):
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
