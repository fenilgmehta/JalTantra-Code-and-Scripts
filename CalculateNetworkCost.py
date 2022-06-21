#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback
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
#   4. AMPL, Baron, Octeract are installed and properly configured
#      (Execution is done from "mtp" directory or any other directory with the same directory structure)
#   5. Model files are present at the right place
#   6. There is no limitation on the amount of available RAM (This assumption make the program simpler.
#      However, in future, it may be removed during deployment to make sure the solvers runs at optimal
#      speed - accordingly changes need to be done in this program)
#   7. Satisfy RegEx r'(a-zA-Z0-9_ )+' -> Absolute path of this Python script, and
#                                         absolute path to graph/network (i.e. data/testcase file)

# ---

# True if STDOUT and STDERR are connected to terminal (and NOT "a file or a pipe")
# REFER: https://github.com/alttch/neotermcolor
# REFER: https://stackoverflow.com/questions/1077113/how-do-i-detect-whether-sys-stdout-is-attached-to-terminal-or-not
# NOTE: This checking of whether stdout and stderr are mapped to terminal or not is required to prevent some garbage
#       values from inadvertently replacing starting lines of the log file to which stdout and stderr are redirected
g_STD_OUT_ERR_TO_TERMINAL = (sys.stdout.isatty() and sys.stderr.isatty())

# REFER: https://stackoverflow.com/questions/3172470/actual-meaning-of-shell-true-in-subprocess
g_BASH_PATH = subprocess.check_output(['which', 'bash'], shell=False).decode().strip()


def run_command(cmd: str, default_result: str = '', debug_print: bool = False) -> Tuple[bool, str]:
    """Execute `cmd` using bash

    `stderr` is merged with `stdout`

    Returns:
        Tuple of [ok, output]
    """
    if debug_print:
        g_logger.debug(f'COMMAND:\n`{cmd}`')
    try:
        # NOTE: Not using the below lines of code because they use "sh" shell,
        #       and it does not seem to properly parse the command.
        #           >>> status_code, output = subprocess.getstatusoutput(cmd)
        #           >>> os.system(...)  # Has the same issue because it uses `sh`
        #   Example 1: `$ kill -s SIGINT 12345` did not work and gave the following error:
        #                  '/bin/sh: 1: kill: invalid signal number or name: SIGINT'
        #              The error logs of testing has been put in "REPO/logs/2022-01-22_ssh_kill_errors.txt"
        #   Example 2: Run the below command in `bash`, `zsh` and `sh` to see the difference in behaviour
        #              `$ echo cool 1>> /dev/stderr &> /dev/null`
        #                  bash --> no output
        #                  zsh  --> output="cool", unexpected behaviour is seen when redirecting stdout to stderr, and
        #                                          unexpected behaviour has been seen with few other commands as well
        #                  sh   --> output="cool", the command is launched asynchronously due of misinterpretation of &>
        output = subprocess.check_output(
            [g_BASH_PATH, '-c', cmd],
            stderr=subprocess.STDOUT,
            shell=False
        ).decode().strip()
        if debug_print:
            g_logger.debug(f'OUTPUT:\n{output}')
        return True, output
    except subprocess.CalledProcessError as e:
        g_logger.info(f'EXCEPTION OCCURRED, will return default_result ("{default_result}") as the output')
        g_logger.info(f'CalledProcessError = {str(e).splitlines()[-1]}')
        g_logger.info(f'Output = {e.output}')
        # g_logger.warning(e)
        # g_logger.warning(traceback.format_exc())
    if debug_print:
        g_logger.debug(default_result)
    return False, default_result


def run_command_get_output(cmd: str, default_result: str = '0', debug_print: bool = False) -> str:
    """Execute `cmd` using bash

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
    """Delete `n` number of lines from stdout. NOTE: This works only if stdout is mapped to a TTY."""
    # REFER: https://www.quora.com/How-can-I-delete-the-last-printed-line-in-Python-language
    for _ in range(n):
        sys.stdout.write('\x1b[1A')  # Cursor up one line
        sys.stdout.write('\x1b[2K')  # Erase line
    sys.stdout.flush()


def get_free_ram() -> float:
    """Returns: free RAM in GiB"""
    # REFER: https://stackoverflow.com/questions/34937580/get-available-memory-in-gb-using-single-bash-shell-command/34938001
    return float(run_command_get_output(r'''awk '/MemFree/ { printf "%.3f\n", $2/1024/1024 }' /proc/meminfo'''))


def get_free_swap() -> float:
    """Returns: free Swap in GiB"""
    # REFER: https://stackoverflow.com/questions/34937580/get-available-memory-in-gb-using-single-bash-shell-command/34938001
    return float(run_command_get_output(r'''awk '/SwapFree/ { printf "%.3f\n", $2/1024/1024 }' /proc/meminfo'''))


def get_execution_time(pid: Union[int, str]) -> int:
    """Returns: wall clock based execution time in seconds"""
    # NOTE: etime measures "wall clock time", i.e. difference between now and the moment the process was started
    #       REFER: https://stackoverflow.com/questions/17737531/linux-ps-command-get-process-running-time-different-between-etime-and-time-p
    # REFER: https://unix.stackexchange.com/questions/7870/how-to-check-how-long-a-process-has-been-running
    return int(run_command(f'ps -o etimes= -p "{pid}"', str(10 ** 15))[1])  # 10**15 seconds == ~3.17 crore years


def get_process_running_status(pid: Union[int, str]) -> bool:
    """Returns: Whether the process with PID=pid is running or not"""
    return run_command(f'ps -p {pid}')[0]


def file_hash_sha256(file_path) -> str:
    """
    This is same as `shasum -a 256 FilePath`. REFER: https://en.wikipedia.org/wiki/Secure_Hash_Algorithms

    It is assumed that the file will exist
    """
    # REFER: https://stackoverflow.com/questions/16874598/how-do-i-calculate-the-md5-checksum-of-a-file-in-python
    total_bytes = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256()
        chunk = f.read(8192)
        total_progress = min(len(chunk), total_bytes)
        last_progress_printed = -1
        if g_STD_OUT_ERR_TO_TERMINAL:
            g_logger.info('--- blank link to handle first use of `delete_last_lines()` '
                          'that happens before first progress logging ---')
        while chunk:
            file_hash.update(chunk)
            new_progress = (100 * total_progress) // total_bytes
            if new_progress != last_progress_printed:
                if g_logger.getEffectiveLevel() <= logging.INFO:
                    if g_STD_OUT_ERR_TO_TERMINAL:
                        delete_last_lines()
                    g_logger.info(f'Hash calculation {new_progress}% done')
                last_progress_printed = new_progress
            chunk = f.read(8192)
            total_progress = min(total_progress + len(chunk), total_bytes)
    return file_hash.hexdigest()


# ---

class SolverOutputAnalyzerParent:
    def __init__(self, engine_path: str, engine_options: str, process_name_to_stop_using_ctrl_c: str):
        """
        Perform output analysis (i.e. extract solution, error messages and other necessary information) for various solvers

        Args:
            engine_path: Path to the solver that will be used by AMPL
            engine_options: Solver specific parameters in AMPL format
            process_name_to_stop_using_ctrl_c: Name of the process that is to be stopped using
                                               Ctrl+C (i.e. SIGINT signal) such that solver smartly
                                               gives us the best solution found till that moment
        """
        self.engine_path = engine_path
        self.engine_options = engine_options
        self.process_name_to_stop_using_ctrl_c = process_name_to_stop_using_ctrl_c

    def check_solution_found(self, exec_info: 'NetworkExecutionInformation') -> bool:
        """
        Parses the output (stdout and stderr) of the solver and tells us
        whether the solver has found any feasible solution or not

        Args:
            exec_info: NetworkExecutionInformation object having all information regarding the execution of the solver

        Returns:
             A boolean value telling whether the solver found any feasible solution or not
        """
        g_logger.error(f"`self.check_solution_found` is 'Not Implemented' for {self.engine_path=}")
        return True

    def extract_best_solution(self, exec_info: 'NetworkExecutionInformation') -> Tuple[bool, float]:
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
        g_logger.error(f"`self.extract_best_solution` is 'Not Implemented' for {self.engine_path=}")
        return True, 0.0

    def check_errors(self, exec_info: 'NetworkExecutionInformation') -> Tuple[bool, str]:
        """By default, it is assumed that everything is ok (i.e. error free)"""
        g_logger.error(f"`self.check_errors` is 'Not Implemented' for {self.engine_path=}")
        return True, '?'

    def extract_solution_vector(self, exec_info: 'NetworkExecutionInformation') -> Tuple[bool, str, float, str]:
        """Returns: Tuple(status, file_to_parse, objective_value, solution_vector)"""
        g_logger.error(f"`self.extract_solution_vector` is 'Not Implemented' for {self.engine_path=}")
        return False, '?', float('nan'), '?'

    @staticmethod
    def ampl_check_errors(file_txt: str) -> Tuple[bool, str]:
        """
        Check if any error has occurred due to AMPL

        Args:
            file_txt: std_out_err.txt file content

        Returns:
            A boolean value telling whether everything is ok (True) or not (False)
            A string value containing the error message
        """
        # AMPL binary does not exist
        try:
            err_idx = file_txt.index('no such file or directory: ./ampl.linux-intel64/ampl')
            err_msg = file_txt.strip()
            g_logger.debug(err_msg)
            return False, err_msg
        except ValueError:
            pass  # substring not found

        # Execute permission not available for AMPL
        try:
            err_idx = file_txt.index('permission denied: ./ampl.linux-intel64/ampl')
            err_msg = file_txt.strip()
            g_logger.debug(err_msg)
            return False, err_msg
        except ValueError:
            pass  # substring not found

        # Execute permission not available for Solver
        try:
            if 'Cannot invoke' in file_txt and 'Permission denied' in file_txt:
                err_idx = file_txt.index('Permission denied')
                err_msg = file_txt[:err_idx + len('Permission denied') + 1].strip()
                g_logger.debug(err_msg)
                return False, err_msg
        except Exception as e:
            g_logger.error(f'FIXME: {type(e)}:\n{e}')

        # AMPL demo licence limitation
        err_idx = None
        try:
            err_idx = file_txt.index('Sorry, a demo license for AMPL is limited to')
            err_msg = file_txt[err_idx:file_txt.index('ampl:', err_idx)].replace('\n', ' ').strip()
            g_logger.debug(err_msg)
            return False, err_msg
        except ValueError:
            if err_idx is not None:
                g_logger.debug(file_txt[err_idx:])
                return False, file_txt[err_idx:]
            pass  # substring not found

        # Probable explanation: some error occurred before the presolve phase
        err_idx = None
        try:
            err_idx = file_txt.index('Error executing "solve" command:')
            err_msg = file_txt[err_idx:file_txt.index('<BREAK>', err_idx)].replace('\n', ' ').strip()
            g_logger.debug(err_msg)
            return False, err_msg
        except ValueError:
            if err_idx is not None:
                g_logger.debug(file_txt[err_idx:])
                return False, file_txt[err_idx:]
            pass  # substring not found
        del err_idx

        # Solver did not proceed due to some impossible situations, e.g. X should be <= 3 and >= 7
        try:
            if '_total_solve_time = 0' in file_txt:
                err_idx = file_txt.index('presolve:')
                err_msg = file_txt[err_idx: file_txt.index('_total_solve_time')]
                g_logger.debug(err_msg)
                return False, err_msg
        except ValueError:
            pass  # substring not found

        return True, 'No Errors'


class SolverOutputAnalyzerBaron(SolverOutputAnalyzerParent):
    pass
    r"""
    ### Baron - v21.1.13

    ```sh
    console
      # Refer the analysis of `sum.lst` for things before execution of Control-C
      BARON: Cntrl-C Abort
      BARON 21.1.13 (2021.01.13): Interrupted by Control-C; objective (numberRegex)
      Retaining scratch directory "/tmp/baron_tmp15378".

    /tmp/baron_tmp15378/sum.lst - Exact copy of console output (output after ctrl+c is not present in this file)
      67: Doing local search
      68: Preprocessing found feasible solution with value  (numberRegex)
      71: Estimated remaining time for local search is [0-9]+ secs
      72: Estimated remaining time for local search is [0-9]+ secs
      73: Done with local search
      75:  Iteration    Open nodes         Time (s)    Lower bound      Upper bound

    /tmp/baron_tmp15378/res.lst - Detailed output / logging of the execution and its status
      67: Doing local search
      68: >>> Preprocessing found feasible solution
      69: >>> Objective value is:           (numberRegex)
      6718:The best solution found is:
      9659:The above solution has an objective value of:  ([0-9]+|([0-9]+)?\.[0-9]|[0-9]+(e|E)(\+)?[0-9]+)

    /tmp/baron_tmp15378/amplmodel.bar - lower bounds, upper bounds, constraints and objective (minimize/maximize) value/expression/equation/inequality with values substituted

    /tmp/baron_tmp15378/dictionary.txt - I did not understand much
    ```
    """

    def __init__(self, engine_path: str, engine_options: str, threads: int):
        process_name_to_stop_using_ctrl_c = 'baron'  # For 1 core and multi core, same process is to be stopped
        super().__init__(engine_path, engine_options, process_name_to_stop_using_ctrl_c)

    def __baron_extract_output_table(self, std_out_err_file_path: str) -> str:
        return run_command_get_output(f"bash output_table_extractor_baron.sh '{std_out_err_file_path}'", '')

    def extract_best_solution(self, exec_info: 'NetworkExecutionInformation') -> Tuple[bool, float]:
        # Extract the solution from the std_out_err file using the value printed by the AMPL commands:
        #     option display_precision 0;
        #     display total_cost;
        try:
            file_txt = open(exec_info.uniq_std_out_err_file_path, 'r').read()
            best_solution = re.search(r'^total_cost\s*=\s*(.*)$', file_txt, re.M).group(1)
            best_solution = float(best_solution)
            ok = True
            if best_solution == 0 or best_solution > 1e40:
                g_logger.warning(f"Probably an infeasible solution found by Baron: '{best_solution}'")
                g_logger.info(f'Instance={exec_info}')
                ok = False
            return ok, best_solution
        except Exception as e:
            g_logger.error(f'CHECKME: {type(e)}, error:\n{e}')
            g_logger.debug('Probably, Baron did not terminate immediately even after receiving the appropriate signal')

        g_logger.info('Using fallback mechanism to extract the best solution')

        csv = self.__baron_extract_output_table(exec_info.uniq_std_out_err_file_path)
        if csv == '':
            return False, 0.0
        lines = csv.split('\n')
        lines = [line for line in lines if line != ',' and (not line.startswith('Processing file'))]
        g_logger.debug(f'{lines=}')

        ok = len(lines) > 0
        best_solution = 0.0
        if len(lines) > 0:
            best_solution = float(lines[-1].split(',')[1])
        if best_solution > 1e40:
            g_logger.warning(f"Probably an infeasible solution found by Baron: '{lines[-1]}'")
            g_logger.info(f'Instance={exec_info}')
            ok = False
        return ok, best_solution

    def check_solution_found(self, exec_info: 'NetworkExecutionInformation') -> bool:
        return self.extract_best_solution(exec_info)[0]

    def check_errors(self, exec_info: 'NetworkExecutionInformation') -> Tuple[bool, str]:
        file_txt = open(exec_info.uniq_std_out_err_file_path, 'r').read()

        ok, err_msg = self.ampl_check_errors(file_txt)
        if not ok:
            return ok, err_msg

        try:
            err_idx = file_txt.index('Sorry, a demo license is limited to 10 variables')
            err_msg = file_txt[err_idx:file_txt.index('exit value 1', err_idx)].replace('\n', ' ').strip()
            g_logger.debug(err_msg)
            return False, err_msg
        except ValueError:
            pass  # substring not found

        try:
            err_idx = re.search(r'''Can't\s+find\s+file\s+['"]?.+['"]?''', file_txt).start()
            g_logger.debug(file_txt[err_idx:])
            return False, file_txt[err_idx:]
        except AttributeError as e:
            # re.search returned None
            g_logger.debug(f'{type(e)}: {e}')
            g_logger.debug(f'{exec_info.uniq_std_out_err_file_path=}')
        except Exception as e:
            g_logger.error(f'FIXME: {type(e)}:\n{e}')

        if 'No feasible solution was found' in file_txt:
            return False, 'No feasible solution was found'

        return True, 'No Errors'

    def __baron_extract_solution_file_path(self, exec_info: 'NetworkExecutionInformation') -> Tuple[bool, str]:
        file_txt = open(exec_info.uniq_std_out_err_file_path, 'r').read()
        solution_dir_name = re.search(r'Retaining scratch directory "/tmp/(.+)"\.', file_txt).group(1)
        if solution_dir_name == '':
            return False, 'RegEx search failed'
        return True, (pathlib.Path(exec_info.aes.OUTPUT_DIR_LEVEL_1_DATA) / solution_dir_name / 'res.lst').resolve()

    def extract_solution_vector(self, exec_info: 'NetworkExecutionInformation') -> Tuple[bool, str, float, str]:
        ok, file_to_parse = self.__baron_extract_solution_file_path(exec_info)
        if not ok:
            return False, file_to_parse, float('nan'), 'Failed to extract solution file path'
        file_txt = open(file_to_parse, 'r').read()
        objective_value = 'NotDefined'
        try:
            objective_value = re.search(r'The above solution has an objective value of:(.+)', file_txt).group(1).strip()
            objective_value = float(objective_value)
        except Exception as e:
            g_logger.error(f'Exception e:\n{e}')
            return False, file_to_parse, float('nan'), f"Objective value (='{objective_value}') RegEx search failed " \
                                                       f"(Probably: no feasible solution was found)"
        try:
            # The lines from '^The best solution found is.+' till the End Of File
            # idx = file_txt.index('The best solution found is:')
            idx_start = re.search(r'variable\s*xlo\s*xbest\s*xup', file_txt).end()
            idx_end = re.search(r'The above solution has an objective value of', file_txt).start()
            solution_vector_list = list()
            for line in file_txt[idx_start:idx_end].strip().splitlines(keepends=False):
                vals = line.strip().split()
                solution_vector_list.append(f'{vals[0]}: {vals[2]}')  # Variable Name, Best Value
            return True, file_to_parse, objective_value, '\n'.join(solution_vector_list)
        except AttributeError as e:
            # re.search returned None
            g_logger.debug(f'{type(e)}: {e}')
            g_logger.debug(f'{exec_info.uniq_std_out_err_file_path=}')
            return False, file_to_parse, objective_value, 'Solution vector RegEx search failed'
        except Exception as e:
            g_logger.error(f'FIXME: {type(e)}:\n{e}')
        # noinspection PyUnreachableCode
        return False, file_to_parse, objective_value, 'FIXME: Unhandled unknown case'


class SolverOutputAnalyzerOcteract(SolverOutputAnalyzerParent):
    """
    Perform output analysis (i.e. extract solution, error messages and other necessary information) for Octeract solver

    Please refer to Octeract documentation for details:
        https://docs.octeract.com/
        https://docs.octeract.com/#solver_options
        https://docs.octeract.com/so1001-general_solver_settings
        https://docplayer.net/187454093-Octeract-engine-user-manual-june-12-2020.html
    """

    def __init__(self, engine_path: str, engine_options: str, threads: int):
        # For 1 core, process with name 'octeract-engine' is the be stopped using Control+C
        # For multi core, process with name 'mpirun' is the be stopped using Control+C
        process_name_to_stop_using_ctrl_c = 'octeract-engine'
        if threads > 1:
            process_name_to_stop_using_ctrl_c = 'mpirun'
        super().__init__(engine_path, engine_options, process_name_to_stop_using_ctrl_c)

    def __octeract_extract_output_table(self, std_out_err_file_path: str) -> str:
        return run_command_get_output(f"bash output_table_extractor_octeract.sh '{std_out_err_file_path}'", '')

    def extract_best_solution(self, exec_info: 'NetworkExecutionInformation') -> Tuple[bool, float]:
        # Extract the solution from the std_out_err file using the value printed by the AMPL commands:
        #     option display_precision 0;
        #     display total_cost;
        try:
            file_txt = open(exec_info.uniq_std_out_err_file_path, 'r').read()
            best_solution = re.search(r'^total_cost\s*=\s*(.*)$', file_txt, re.M).group(1)
            best_solution = float(best_solution)
            ok = True
            if best_solution == 0 or best_solution > 1e40:
                g_logger.warning(f"Probably an infeasible solution found by Octeract: '{best_solution}'")
                g_logger.info(f'Instance={exec_info}')
                ok = False
            return ok, best_solution
        except Exception as e:
            g_logger.error(f'CHECKME: {type(e)}, error:\n{e}')
            g_logger.debug('Probably, Octeract did not terminate immediately '
                           'even after receiving the appropriate signal')

        g_logger.info('Using fallback mechanism to extract the best solution')

        # Custom checking for exceptional situations
        file_txt = open(exec_info.uniq_std_out_err_file_path, 'r').read()
        if 'Found solution during preprocessing' in file_txt:
            try:
                best_solution = float(re.search(r'Objective value at global solution:\s*(.*)\s*', file_txt).group(1))
                g_logger.debug(f'Solver found the solution during preprocessing, {best_solution=}')
                return True, best_solution
            except:
                g_logger.error(f'FIXME: Found solution during preprocessing. But, failed to extract the objective '
                               f'function value. Few lines from {exec_info.uniq_std_out_err_file_path=}:')
                idx_found_solution = file_txt.find('Found solution during preprocessing')
                g_logger.debug(file_txt[file_txt.rfind('\n', 0, idx_found_solution - 50) + 1:
                                        file_txt.find('\n', idx_found_solution + 120)])
                pass

        # Use bash script to extract the values from the table printed to output by Octeract
        csv = self.__octeract_extract_output_table(exec_info.uniq_std_out_err_file_path)
        if csv == '':
            return False, 0.0
        lines = csv.split('\n')
        lines = [line for line in lines if line != ',' and (not line.startswith('Processing file'))]
        g_logger.debug(f'{lines=}')

        status = len(lines) > 0
        best_solution = 0.0
        g_logger.info(f'Instance={exec_info}')
        if len(lines) > 0:
            last_line_splitted = lines[-1].split(',')
            if len(last_line_splitted) > 2:
                if last_line_splitted[-1] == '(I)':
                    g_logger.warning(f"Infeasible solution found by Octeract: '{lines[-1]}'")
                    status = False
                else:
                    g_logger.warning(f"FIXME: Unhandled unknown case, probably an infeasible "
                                     f"solution found by Octeract: '{lines[-1]}'")
            else:
                best_solution = float(last_line_splitted[1])
        g_logger.debug((status, best_solution))
        return status, best_solution

    def check_solution_found(self, exec_info: 'NetworkExecutionInformation') -> bool:
        return self.extract_best_solution(exec_info)[0]

    def check_errors(self, exec_info: 'NetworkExecutionInformation') -> Tuple[bool, str]:
        file_txt = open(exec_info.uniq_std_out_err_file_path, 'r').read()

        if 'Found solution during preprocessing' in file_txt:
            return True, 'Solution found during preprocessing'
        if 'Iteration            GAP               LLB          BUB            Pool       Time       Mem' in file_txt:
            return True, 'Probably No Errors'

        ok, err_msg = self.ampl_check_errors(file_txt)
        if not ok:
            return ok, err_msg

        try:
            err_idx = re.search(r'''Can't\s+find\s+file\s+['"]?.+['"]?''', file_txt).start()
            g_logger.debug(file_txt[err_idx:])
            return False, file_txt[err_idx:]
        except IndexError as e:
            g_logger.error(f'{type(e)}: {e}')
            g_logger.debug(f'{exec_info.uniq_std_out_err_file_path=}')
        except AttributeError as e:
            # re.search failed, returned `None`
            g_logger.debug(f'{type(e)}: {e}')
            g_logger.debug(f'{exec_info.uniq_std_out_err_file_path=}')
        except Exception as e:
            g_logger.error(f'FIXME: {type(e)}:\n{e}')

        err_idx = None
        try:
            err_idx = file_txt.index('Request_Error')
            err_msg = file_txt[err_idx:file_txt.index('exit value 1', err_idx)].replace('\n', ' ').strip()
            g_logger.debug(err_msg)
            return False, err_msg
        except ValueError:
            if err_idx is not None:
                g_logger.debug(file_txt[err_idx:])
                return False, file_txt[err_idx:]
            pass  # substring not found

        # Octeract solver failed to establish connection to their server. Hence, it does not proceed
        # with the solving.
        err_idx = None
        try:
            err_idx = file_txt.index('Error: Failed to establish connection to server.')
            if ("----------------------------------------------------------------------"
                "--------------------------" not in file_txt) or "can't open /tmp/at" in file_txt:
                err_msg = file_txt[err_idx:file_txt.index('ampl:', err_idx)].replace('\n', ' ').strip()
                g_logger.debug(err_msg)
                return False, err_msg
            else:
                g_logger.debug(f'CHECKME: {file_txt[err_idx:]=}')
        except ValueError:
            if err_idx is not None:
                g_logger.debug(file_txt[err_idx:])
                return False, file_txt[err_idx:]
            pass  # substring not found

        try:
            err_idx = file_txt.index('presolve messages suppressed')
            if '_total_solve_time' in file_txt:
                err_msg = file_txt[:file_txt.index('_total_solve_time')]
            else:
                err_msg = file_txt[:err_idx + len('presolve messages suppressed') + 1]
            err_msg = err_msg.strip()
            g_logger.debug(err_msg)
            return False, err_msg
        except ValueError:
            pass  # substring not found

        try:
            if 'all variables eliminated, but lower bound' in file_txt:
                err_idx = file_txt.index('_total_solve_time')
                err_msg = file_txt[:err_idx]
                g_logger.debug(err_msg)
                return False, err_msg
        except ValueError:
            pass  # substring not found

        return True, 'No Errors'

    def __octeract_extract_solution_file_path(self, exec_info: 'NetworkExecutionInformation') -> Tuple[bool, str]:
        file_txt = open(exec_info.uniq_std_out_err_file_path, 'r').read()
        solution_file_name = re.search(r'Solution file written to: /tmp/(.+)', file_txt).group(1)
        if solution_file_name == '':
            g_logger.debug(f"FIXME: log file path='{exec_info.uniq_std_out_err_file_path}', file_txt:\n{file_txt}")
            return False, 'RegEx search failed'
        return True, (pathlib.Path(exec_info.aes.OUTPUT_DIR_LEVEL_1_DATA) / solution_file_name).resolve()

    def extract_solution_vector(self, exec_info: 'NetworkExecutionInformation') -> Tuple[bool, str, float, str]:
        ok, file_to_parse = self.__octeract_extract_solution_file_path(exec_info)
        if not ok:
            return False, file_to_parse, float('nan'), ''
        json_data = json.loads(open(file_to_parse, 'r').read())
        if not json_data['feasible']:
            return False, file_to_parse, json_data['objective_value'], json_data['statistics']['dgo_exit_status']
        objective_value: float = json_data['objective_value']
        solution_vector = '\n'.join(
            [f'{i}: {j}' for i, j in sorted(json_data['solution_vector'].items(), key=lambda kv: (len(kv[0]), kv[0]))]
        )
        return True, file_to_parse, objective_value, solution_vector


# ---

class NetworkExecutionInformation:
    def __init__(self, aes: 'AutoExecutorSettings', idx: int):
        """
        This constructor will set all data members except `self.tmux_bash_pid`
        """
        self.tmux_bash_pid: Union[str, None] = None  # This has to be set manually
        self.idx: int = idx
        self.aes: 'AutoExecutorSettings' = aes
        self.solver_name, self.model_name = aes.solver_model_combinations[idx]
        self.solver_info: SolverOutputAnalyzerParent = aes.solvers[self.solver_name]

        # REFER: https://stackoverflow.com/a/66771847
        self.short_uniq_model_name: str = self.model_name[:self.model_name.find('_')]
        # REFER: https://github.com/tmux/tmux/issues/3113
        #        Q. What is the maximum length of session name that can be set using
        #           the following command: `tmux new -s 'SessionName_12345'`
        #        A. There is no defined limit.
        self.short_uniq_data_file_name: str = aes.data_file_hash
        self.short_uniq_combination: str = f'{self.solver_name}_' \
                                           f'{self.short_uniq_model_name}_{self.short_uniq_data_file_name}'
        g_logger.debug((type(self.short_uniq_model_name), type(self.short_uniq_data_file_name),
                        type(self.short_uniq_combination)))
        g_logger.debug((self.short_uniq_model_name, self.short_uniq_data_file_name, self.short_uniq_combination))

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

    def __repr__(self):
        # REFER: https://stackoverflow.com/questions/727761/python-str-and-lists
        # REFER: https://docs.python.org/2/reference/datamodel.html#object.__repr__
        return self.__str__()


# ---

class AutoExecutorSettings:
    # Level 0 is main directory inside which everything will exist
    OUTPUT_DIR_LEVEL_0 = './NetworkResults/'.rstrip('/')  # Note: Do not put trailing forward slash ('/')
    OUTPUT_DIR_LEVEL_1_DATA = f'{OUTPUT_DIR_LEVEL_0}/SolutionData'
    # Please ensure that proper escaping of white spaces and other special characters
    # is done because this will be executed in a fashion similar to `./a.out`
    AMPL_PATH = './ampl.linux-intel64/ampl'
    AVAILABLE_SOLVERS = ['baron', 'octeract']  # NOTE: Also look at `__update_solver_dict()` method when updating this
    AVAILABLE_MODELS = {1: 'm1_basic.R', 2: 'm2_basic2_v2.R', 3: 'm3_descrete_segment.R', 4: 'm4_parallel_links.R'}
    TMUX_UNIQUE_PREFIX = f'AR_NC_{os.getpid()}_'  # AR = Auto Run, NC = Network Cost

    def __init__(self):
        self.debug = False
        self.r_cpu_cores_per_solver = 1
        # 48 core server is being used
        self.r_max_parallel_solvers = 44
        # Time is in seconds, set this to any value <= 0 to ignore this parameter
        self.r_execution_time_limit = (0 * 60 * 60) + (5 * 60) + 0
        self.r_min_free_ram = 2  # GiB
        self.r_min_free_swap = 8  # GiB, usefulness of this variable depends on the swappiness of the system

        self.models_dir = "./Files/Models"  # m1, m3 => q   ,   m2, m4 => q1, q2
        self.solvers: Dict[str, SolverOutputAnalyzerParent] = {}
        self.__update_solver_dict()

        # Tuples of (Solver name & Model name) which are to be executed to
        # find the cost of the given graph/network (i.e. data/testcase file)
        self.solver_model_combinations: List[Tuple[str, str]] = list()
        # Path to graph/network (i.e. data/testcase file)
        self.data_file_path: str = ''
        self.data_file_hash: str = ''
        self.output_dir_level_1_network_specific: str = ''
        self.output_network_specific_result: str = ''
        self.output_result_summary_file: str = ''

    def __update_solver_dict(self):
        # NOTE: Update `AutoExecutorSettings.AVAILABLE_SOLVERS` if keys in below dictionary are updated
        # NOTE: Use double quotes ONLY in the below variables
        self.solvers = {
            'baron': SolverOutputAnalyzerBaron(
                engine_path='./ampl.linux-intel64/baron',
                engine_options=f'option baron_options "threads={self.r_cpu_cores_per_solver} '
                               f'barstats keepsol lsolmsg outlev=1 prfreq=100 prtime=2 problem";',
                threads=self.r_cpu_cores_per_solver
            ),
            'octeract': SolverOutputAnalyzerOcteract(
                engine_path='./octeract-engine-4.0.0/bin/octeract-engine',
                engine_options=f'options octeract_options "num_cores={self.r_cpu_cores_per_solver}";',
                threads=self.r_cpu_cores_per_solver
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
        self.data_file_hash = file_hash_sha256(data_file_path)
        self.output_dir_level_1_network_specific = f'{AutoExecutorSettings.OUTPUT_DIR_LEVEL_0}' \
                                                   f'/{self.data_file_hash}'
        self.output_network_specific_result = self.output_dir_level_1_network_specific + '/0_result.txt'
        self.output_result_summary_file = self.output_dir_level_1_network_specific + '/0_result_summary.txt'

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
        # REFER: https://github.com/fenilgmehta/Jaltantra-Code-and-Scripts/blob/main/Files/main.run
        #        For AMPL commands
        run_command_get_output(rf'''
            tmux new-session -d -s '{info.uniq_tmux_session_name}' '
echo $$ > "{info.uniq_pid_file_path}"
{self.AMPL_PATH} > "{info.uniq_std_out_err_file_path}" 2>&1 <<EOF
    reset;
    model "{info.models_dir}/{info.model_name}";
    data "{info.data_file_path}";
    option solver "{info.engine_path}";
    option presolve_eps 1e-9;
    {info.engine_options};
    solve;
    ''' + r'''
    display _total_solve_time;
    option display_1col 9223372036854775807;
    option display_precision 6;
    display {i in nodes} h[i];
    display {(i,j) in arcs} ''' + ("q[i,j]" if (info.short_uniq_model_name in ("m1", "m3")) else "(q1[i,j], q2[i,j])") + ''';
    option display_eps 1e-4;
    option omit_zero_rows 1;
    display {(i,j) in arcs, k in pipes} l[i,j,k];
    display _total_solve_time;
    option display_precision 0;
    display total_cost;
EOF
echo > /dev/null
'
        ''', debug_print=self.debug)
        # At max we wait for 60 seconds
        pid_file_wait_time = 0
        while (not os.path.exists(info.uniq_pid_file_path)) and (pid_file_wait_time < 60):
            g_logger.info('PID file not generated. Sleeping for 1 second...')
            pid_file_wait_time += 1
            time.sleep(1)
        if os.path.exists(info.uniq_pid_file_path):
            info.tmux_bash_pid = run_command_get_output(f'cat "{info.uniq_pid_file_path}"')
        else:
            g_logger.error(f'FIXME: CHECKME: PID file not created for {info=}')
            info.tmux_bash_pid = 0
        return info


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
    """Return True if a feasible solution has been found by any one of the tmux session (i.e. solver-model combination)"""
    for info in tmux_monitor_list:
        if info.solver_info.check_solution_found(info):
            return True
    return False


def extract_best_solution(
        tmux_monitor_list: List[NetworkExecutionInformation],
        result_summary_file_path: str
) -> Tuple[bool, float, Optional[NetworkExecutionInformation]]:
    """
    Args:
        tmux_monitor_list: List of `NetworkExecutionInformation` which have finished their execution

    Returns:
        ok, best solution, context of solver and model which found the best solution
    """
    all_results: List = list()  # Only used for debugging
    best_result_till_now, best_result_exec_info = float('inf'), None
    for exec_info in tmux_monitor_list:
        ok, curr_res = exec_info.solver_info.extract_best_solution(exec_info)
        all_results.append((exec_info.solver_name, exec_info.short_uniq_model_name, ok, curr_res))
        g_logger.info(f'solver={exec_info.solver_name}, model={exec_info.short_uniq_model_name}, {ok=}, {curr_res=}')
        # `if` solution not found by this solver instance `or` a better solution is already known, then `continue`
        if not ok or curr_res >= best_result_till_now:
            continue
        g_logger.debug(f'Update best result seen till now: {curr_res} < {best_result_till_now=}')
        best_result_till_now = curr_res
        best_result_exec_info = exec_info
    for (solver_name, model_name, ok, res) in all_results:
        g_logger.info(f'solver={solver_name}, model={model_name}, {ok=}, {res=}')
        run_command(f"echo 'solver={solver_name}, model={model_name}, {ok=}, {res=}'"
                    f" >> '{result_summary_file_path}'")
    return best_result_exec_info is not None, best_result_till_now, best_result_exec_info


def main(my_settings: AutoExecutorSettings) -> None:
    """Launch the solvers and monitor them based on CLI parameters"""
    # Create the required directory structure
    run_command_get_output(f'mkdir -p "{my_settings.OUTPUT_DIR_LEVEL_0}"')
    run_command_get_output(f'mkdir -p "{my_settings.OUTPUT_DIR_LEVEL_1_DATA}"')
    run_command_get_output(f'mkdir -p "{my_settings.output_dir_level_1_network_specific}"')

    # Create hardlink to the file passed using -p/--path parameter
    run_command(
        f"ln '{my_settings.data_file_path}' '{(pathlib.Path(my_settings.output_dir_level_1_network_specific) / '0_graph_network_data_testcase.R').resolve()}'"
    )

    # Write metadata to "0_metadata" file
    # REFER: https://www.geeksforgeeks.org/how-to-convert-python-dictionary-to-json/
    g_logger.info('START: Writing requests metadata to 0_metadata file')
    current_time = datetime.datetime.now()  # REFER: https://www.geeksforgeeks.org/get-current-timestamp-using-python/
    metadata_json_str = json.dumps(
        dict(
            unique_prefix=my_settings.TMUX_UNIQUE_PREFIX,
            start_time=str(current_time),
            start_timestamp=current_time.timestamp(),
            solver_execution_time_limit_in_seconds=my_settings.r_execution_time_limit,
            solver_cpu_cores=my_settings.r_cpu_cores_per_solver,
        ),
        indent=2
    )
    with open(f'{my_settings.output_dir_level_1_network_specific}/0_metadata', 'w') as metadata_file:
        metadata_file.write(metadata_json_str)
    del current_time, metadata_json_str, metadata_file
    g_logger.info('FINISHED: Writing requests metadata to 0_metadata file')

    tmux_original_list: List[NetworkExecutionInformation] = list()
    tmux_monitor_list: List[NetworkExecutionInformation] = list()
    tmux_finished_list: List[NetworkExecutionInformation] = list()

    min_combination_parallel_solvers = min(
        len(my_settings.solver_model_combinations),
        my_settings.r_max_parallel_solvers
    )

    # Begin execution of first batch
    g_logger.info('START: Execution of first batch of solvers')
    run_command(f"echo 'running' > {my_settings.output_dir_level_1_network_specific}/0_status")
    for i in range(min_combination_parallel_solvers):
        exec_info = my_settings.start_solver(i)
        g_logger.debug(str(exec_info))
        g_logger.debug(f'{exec_info.tmux_bash_pid=}')
        if exec_info.tmux_bash_pid == '0':
            g_logger.error('FIXME: exec_info.tmux_bash_pid is 0')
            continue
        tmux_original_list.append(exec_info)
        tmux_monitor_list.append(exec_info)
        g_logger.info(f'tmux session "{exec_info.short_uniq_combination}" -> {exec_info.tmux_bash_pid}')
        time.sleep(0.2)
        g_logger.debug(f'{len(tmux_monitor_list)=}')
    del i, exec_info
    g_logger.info('FINISHED: Execution of first batch of solvers')

    # Error checking - Round 1
    # This is done to find and log the errors that occurred just after launching the tmux sessions
    g_logger.info('START: Error checking - Round 1')
    tmux_monitor_list_idx_to_remove = list()
    for idx, exec_info in enumerate(tmux_monitor_list):
        if get_process_running_status(exec_info.tmux_bash_pid):
            continue
        g_logger.warning(f'tmux session stopped "{exec_info.short_uniq_combination}" -> {exec_info.tmux_bash_pid}')
        g_logger.error(exec_info.solver_info.check_errors(exec_info))
        tmux_monitor_list_idx_to_remove.append(idx)
    for idx in reversed(tmux_monitor_list_idx_to_remove):
        tmux_finished_list.insert(0, tmux_monitor_list.pop(idx))
    del tmux_monitor_list_idx_to_remove, idx, exec_info
    g_logger.info('FINISHED: Error checking - Round 1')

    # If all (tmux) sessions of the first batch stopped because of some error, then exit the program
    g_logger.debug(f'{tmux_monitor_list=}')
    if len(tmux_monitor_list) == 0:
        g_logger.warning('Failed to start all solver model sessions')

        # Sleep for some time so that AMPL and the solver can finish their termination properly, and write the
        # appropriate error messages to STDOUT (which are redirected to "Solver_m1_NetworkHash/std_out_err.txt")
        pass
        # TODO: Probably, the below 3 lines can be replaced with the 4 lines after them
        g_logger.info('Sleep for 20 seconds for AMPL and the solver to terminate properly')
        time.sleep(20)
        run_command(f"echo 'launch_error' > {my_settings.output_dir_level_1_network_specific}/0_status")
        # g_logger.debug(run_command(f"while [[ $(tmux ls 2> /dev/null | grep "{g_settings.TMUX_UNIQUE_PREFIX}" | wc -l) > 0 ]] ; do echo 'tmux sessions still running, sleeping for 1 second' ; sleep 1 ; done"))
        # # g_logger.info('Sleep for 20 seconds for AMPL and the solver to terminate properly')
        # # time.sleep(20)
        # run_command(f"echo 'launch_error' > {g_settings.output_dir_level_1_network_specific}/0_status")
        pass

        for exec_info in tmux_finished_list:
            ok, msg = exec_info.solver_info.check_errors(exec_info)
            if ok:
                continue
            # REFER: https://stackoverflow.com/questions/1250079/how-to-escape-single-quotes-within-single-quoted-strings
            msg = msg.replace("'", "'\"'\"'")
            run_command(f"echo '\n---+++---\n\n{exec_info.solver_name}, {exec_info.short_uniq_model_name}'"
                        f" >> {my_settings.output_dir_level_1_network_specific}/0_status")
            run_command(f"echo '\n{msg}' >> {my_settings.output_dir_level_1_network_specific}/0_status")
        exit(3)

    # TODO: Problem: Handle case of deadlock like situation
    #         1. `g_settings.solver_model_combinations > g_settings.MAX_PARALLEL_SOLVERS`
    #         2. The first `g_settings.MAX_PARALLEL_SOLVERS` solver model combinations are poor and
    #            the solver is unable to find any feasible solution even after executing for hours
    #       Solution 1: Add a flag to impose hard deadline on execution time, i.e. the execution of a solver is
    #                   to be stopped if no solution is found by it within the specified hard deadline timelimit
    #       NOTE: Partially fixed in commit ef1901faa5b9b6d8411df0f8625791a38c851527
    # TODO: Problem: See if we can optimise the execution in the below situation:
    #         1. `g_settings.solver_model_combinations > g_settings.MAX_PARALLEL_SOLVERS`
    #         2. Some error occur in the execution of a one of
    #            `g_settings.solver_model_combinations[:g_settings.MAX_PARALLEL_SOLVERS]`
    #            way before `g_settings.EXECUTION_TIME_LIMIT`
    execution_time_left = my_settings.r_execution_time_limit
    while execution_time_left >= 5:
        g_logger.debug(f'Time Finished = '
                       f'{str(datetime.timedelta(seconds=my_settings.r_execution_time_limit - execution_time_left))}'
                       f', Time Left = {str(datetime.timedelta(seconds=execution_time_left))}')
        g_logger.debug(
            "Tmux session count = " +
            run_command_get_output(f'tmux ls 2> /dev/null | grep "{my_settings.TMUX_UNIQUE_PREFIX}" | wc -l')
        )
        if run_command(f'tmux ls 2> /dev/null | grep "{my_settings.TMUX_UNIQUE_PREFIX}" | wc -l', '0')[1] == '0':
            g_logger.info(f'{execution_time_left=}')
            g_logger.info('CHECKME: Skipping the sleep/wait operation as no tmux session is '
                          'running, probably some error or the solver(s) exited early')
            tmux_finished_list.extend(tmux_monitor_list)
            tmux_monitor_list.clear()
            break
        time.sleep(5)
        execution_time_left -= 5
        if my_settings.debug and g_STD_OUT_ERR_TO_TERMINAL:
            delete_last_lines(2)
    if execution_time_left <= 5:
        time.sleep(execution_time_left)
        g_logger.info(f'Initial time limit over '
                      f'(i.e. {str(datetime.timedelta(seconds=my_settings.r_execution_time_limit))})')
        g_logger.debug(
            "Tmux session count = " +
            run_command_get_output(f'tmux ls 2> /dev/null | grep "{my_settings.TMUX_UNIQUE_PREFIX}" | wc -l')
        )
    else:
        g_logger.info('while loop forcefully stopped using `break` as no `tmux` session were running')
    del execution_time_left

    # Error checking - Round 2
    # This is done primarily for logging purpose
    g_logger.info('START: Error checking - Round 2')
    for exec_info in tmux_original_list:
        ok, err_msg = exec_info.solver_info.check_errors(exec_info)
        if not ok:
            g_logger.warning(str(exec_info))
            g_logger.error(err_msg)
    del exec_info, ok, err_msg
    g_logger.info('FINISHED: Error checking - Round 2')

    # - Check if any feasible solution has been found by the first batch of tmux sessions or not.
    # - If no feasible solution has been found by any of the session of the first batch, then give them
    #   limited amount of extra time to find a feasible solution.
    # - If "a feasible solution is found" or "extra time has been consumed", then proceed with the second batch.
    # NOTE: `at_least_one_solution_found` variable is unused as of now. However, it may be needed in future.
    at_least_one_solution_found = False
    extra_time_given = 0
    while len(tmux_monitor_list) > 0 and (not check_solution_status(tmux_monitor_list)):
        extra_time_given += 30
        g_logger.info(f'Extra time given = {extra_time_given} of 300 seconds')
        time.sleep(30)  # Give 30 more seconds to the running solvers
        # TODO: NOTE: In future, this too can be taken as a command line parameter
        # Maximum 5 minutes extra time is given
        if extra_time_given >= 300:
            g_logger.info('"Extra time" limit reached')
            break
    # NOTE: The below statement will work in most cases. However, there can be corner cases where a solution
    #       is found by one of the session of first batch but still `at_least_one_solution_found` is False.
    #       This case can occur when a solver finds the solution at the time between "checking of solution status
    #       in the above loop" and "stopping of that session in the below `for` loop of second batch"
    at_least_one_solution_found = (extra_time_given < 300)
    del extra_time_given, at_least_one_solution_found

    # Begin execution of the remaining solver-model combinations
    g_logger.info('START: Execution of the remaining solver-model combinations')
    for i in range(min_combination_parallel_solvers, len(my_settings.solver_model_combinations)):
        g_logger.debug(run_command_get_output(f'tmux ls | grep "{my_settings.TMUX_UNIQUE_PREFIX}"', debug_print=True))

        while True:
            tmux_sessions_running = int(run_command_get_output(
                f'tmux ls 2> /dev/null | grep "{my_settings.TMUX_UNIQUE_PREFIX}" | wc -l'
            ))
            g_logger.debug(tmux_sessions_running)

            if tmux_sessions_running < my_settings.r_max_parallel_solvers:
                break
            g_logger.debug("----------")
            g_logger.debug(f'{tmux_monitor_list=}')
            g_logger.debug(f'{len(tmux_finished_list)=}')
            MonitorAndStopper.mas_time(tmux_monitor_list, tmux_finished_list, my_settings.r_execution_time_limit, True)
            g_logger.debug(f'{tmux_monitor_list=}')
            g_logger.debug(f'{len(tmux_finished_list)=}')

        exec_info = my_settings.start_solver(i)
        tmux_original_list.append(exec_info)
        tmux_monitor_list.append(exec_info)
        g_logger.info(f'tmux session "{exec_info.short_uniq_combination}" -> {exec_info.tmux_bash_pid}')
        time.sleep(0.2)
        del tmux_sessions_running, exec_info
    g_logger.info('FINISHED: Execution of the remaining solver-model combinations')

    # NOTE: The below loop is required to
    #       1. wait for the tmux sessions specific to the current PID to stop (either naturally before the
    #          execution time limit, or get forcefully stopped if execution time limit is exceeded)
    #       2. move finished tmux session from monitor list to finished list
    while len(tmux_monitor_list):
        g_logger.debug("----------")
        g_logger.debug(f'{tmux_monitor_list=}')
        g_logger.debug(f'{len(tmux_finished_list)=}')
        MonitorAndStopper.mas_time(tmux_monitor_list, tmux_finished_list, my_settings.r_execution_time_limit, True)
        g_logger.debug(f'{tmux_monitor_list=}')
        g_logger.debug(f'{len(tmux_finished_list)=}')

    # Error checking - Round 3 (last round)
    # This is done primarily for logging purpose
    g_logger.info('START: Error checking - Round 3 (last round)')
    for exec_info in tmux_original_list:
        ok, err_msg = exec_info.solver_info.check_errors(exec_info)
        if not ok:
            g_logger.warning(str(exec_info))
            g_logger.error(err_msg)
    del exec_info, ok, err_msg
    g_logger.info('FINISHED: Error checking - Round 3 (last round)')

    # NOTE: We will not copy files with glob `/tmp/at*nl` because they
    #       are only used to pass information from AMPL to solver
    g_logger.info('START: Copying solution files from /tmp to `g_settings.OUTPUT_DIR_LEVEL_1_DATA`')
    run_command_get_output(f"cp -r /tmp/at*octsol /tmp/baron_tmp* '{my_settings.OUTPUT_DIR_LEVEL_1_DATA}'")
    g_logger.info('FINISHED: Copying solution files from /tmp to `g_settings.OUTPUT_DIR_LEVEL_1_DATA`')

    # - Wait for the `tmux` sessions to truly end. This is required because, `MonitorAndStopper.mas_time`
    #   only sends the SIGINT signal to the solvers. The solvers start their stopping process after receiving
    #   this SIGINT signal. But, they do not stop immediately (one possible reason what was found in Octeract
    #   was that it tries to "communicate the results to its servers" and keeps retrying for "some limited
    #   number of times" if "the server is unreachable due to some reason").
    # - If this loop is not there, then there is a possibility that we think that the solver-model combination
    #   failed even if solver had found a "feasible or global" solution, because the solver had not returned the
    #   final value of the variables that were to be calculated/found to AMPL for printing (here, to std_out_err.txt)
    while run_command(f'tmux ls 2> /dev/null | grep "{my_settings.TMUX_UNIQUE_PREFIX}" | wc -l', '0', True)[1] != '0':
        g_logger.info('WAITING for the tmux instances to stop (probably AMPL or some solver has not terminated)')
        time.sleep(10)

    g_logger.info('START: Extracting best solution among all solver-model instances')
    g_logger.debug(f'{len(tmux_monitor_list)=}')
    g_logger.debug(f'{len(tmux_finished_list)=}')
    status, best_cost, best_cost_instance_exec_info = extract_best_solution(tmux_original_list,
                                                                            my_settings.output_result_summary_file)
    g_logger.debug((status, best_cost, str(best_cost_instance_exec_info)))
    g_logger.info('FINISHED: Extracting best solution among all solver-model instances')

    # Do not overwrite the result file of previous execution
    # Just rename the result file of previous execution
    if os.path.exists(my_settings.output_network_specific_result):
        path_prefix, path_suffix = os.path.splitext(my_settings.output_network_specific_result)
        new_path = None
        for i in range(1, 10_000_000):
            new_path = f'{path_prefix}_{i:07}{path_suffix}'
            if not os.path.exists(new_path):
                break
        os.rename(src=my_settings.output_network_specific_result, dst=new_path)
        g_logger.info(f"Old Solution Renamed from '{os.path.split(my_settings.output_network_specific_result)[1]}'"
                      f" -> '{os.path.split(new_path)[1]}'")
        del path_prefix, path_suffix, new_path, i

    g_logger.info('START: Writing the best solution found to result file')
    if not status:
        g_logger.error('NO feasible solution found')
        run_command(f"echo '{status}' > '{my_settings.output_network_specific_result}'")
        run_command(f"echo 'finished:Either some unknown error, or NO feasible solution found'"
                    f" > {my_settings.output_dir_level_1_network_specific}/0_status")
        for exec_info in tmux_finished_list:
            ok, msg = exec_info.solver_info.check_errors(exec_info)
            if ok:
                continue
            msg = msg.replace("'", "'\"'\"'")
            run_command(f"echo '\n---+++---\n\n{exec_info.solver_name}, {exec_info.short_uniq_model_name}'"
                        f" >> {my_settings.output_dir_level_1_network_specific}/0_status")
            run_command(f"echo '\n{msg}' >> {my_settings.output_dir_level_1_network_specific}/0_status")
        del exec_info, ok, msg
        g_logger.info('FINISHED: Writing the best solution found to result file')
        return

    run_command(f"echo 'success' > {my_settings.output_dir_level_1_network_specific}/0_status")
    # status, file_to_parse, objective_value, solution_vector = \
    #     best_cost_instance_exec_info.solver_info.extract_solution_vector(best_cost_instance_exec_info)
    g_logger.info(f'{best_cost=}')
    g_logger.info(f'Instance={best_cost_instance_exec_info}')
    g_logger.info(f'Solver={best_cost_instance_exec_info.solver_name}, '
                  f'Model={best_cost_instance_exec_info.short_uniq_model_name}')
    # Line 1 = Status (success => True, failure => False)
    run_command(f"echo '{True}' > '{my_settings.output_network_specific_result}'")
    # Line 2 = Solver Name
    run_command(f"echo '{best_cost_instance_exec_info.solver_name}' >> '{my_settings.output_network_specific_result}'")
    # Line 3 = Model Name (unique short form)
    run_command(f"echo '{best_cost_instance_exec_info.short_uniq_model_name}'"
                f" >> '{my_settings.output_network_specific_result}'")
    # Line 4 = std_out_err file path of Solver-Model combination
    run_command(f"echo '{best_cost_instance_exec_info.uniq_std_out_err_file_path}'"
                f" >> '{my_settings.output_network_specific_result}'")
    # Line 5 = Best value of the objective function that was found by the Solver
    run_command(f"echo '{best_cost}' >> '{my_settings.output_network_specific_result}'")
    # Line 6+ = Solution data extracted from std_out_err file of the best Solver-Model combination
    #           using `CalculateNetworkCost_ExtractResultFromAmplOutput.py` program
    run_command(f"'{sys.executable}' 'CalculateNetworkCost_ExtractResultFromAmplOutput.py' "
                f"'{best_cost_instance_exec_info.uniq_std_out_err_file_path}' "
                f"'{(pathlib.Path(my_settings.output_dir_level_1_network_specific) / '0_graph_network_data_testcase.R').resolve()}' "
                f"1"
                f" >> '{my_settings.output_network_specific_result}'")
    # run_command(f"echo '{file_to_parse}' >> '{my_settings.output_network_specific_result}'")  # Line 6
    # run_command(f"echo '{objective_value}' >> '{my_settings.output_network_specific_result}'")  # Line 7
    # run_command(f"echo '{solution_vector}' >> '{my_settings.output_network_specific_result}'")  # Line 8+

    g_logger.info('FINISHED: Writing the best solution found to result file')
    return


def update_settings(args: argparse.Namespace) -> AutoExecutorSettings:
    """
    Store the settings in my_settings object and return it

    Args:
        args: command line arguments parsed by `argparse`
    """
    global g_logger

    my_settings = AutoExecutorSettings()
    my_settings.debug = args.debug

    # REFER: https://github.com/alttch/neotermcolor/blob/master/neotermcolor/__init__.py#L120
    # If STDOUT and STDERR are connected to a terminal, then use `rich` logging, otherwise simple logging
    if (os.getenv('ANSI_COLORS_DISABLED') is None) and (sys.stdout.isatty() and sys.stderr.isatty()):
        # noinspection PyArgumentList
        logging.basicConfig(
            level=(logging.DEBUG if args.debug else logging.WARNING),
            format='%(funcName)s :: %(message)s',
            datefmt="[%X]",
            handlers=[rich_RichHandler()]
        )
        g_logger = logging.getLogger('CNC')
    else:
        g_logger = logging.getLogger('CNC')
        if args.debug:
            g_logger.setLevel(logging.DEBUG)
        else:
            g_logger.setLevel(logging.WARNING)
        logger_file_handler = logging.FileHandler('/dev/stderr')
        logger_formatter = logging.Formatter('%(levelname)s :: [%(lineno)s] %(name)s.%(funcName)s :: %(message)s')
        logger_file_handler.setFormatter(logger_formatter)
        g_logger.addHandler(logger_file_handler)
        del logger_file_handler, logger_formatter

    g_logger.debug(args)

    # Check if the network file exists or not
    if not os.path.exists(args.path):
        g_logger.error(f"Cannot access '{args.path}': No such file or directory")
        exit(2)
    g_logger.info(f"Current working directory = '{os.getcwd()}'")
    my_settings.set_data_file_path(args.path)
    g_logger.info(f"Graph/Network (i.e. Data/Testcase file) = '{my_settings.data_file_path}'")
    g_logger.info(f"Input file hash = '{my_settings.data_file_hash}'")

    my_settings.set_execution_time_limit(seconds=args.time)
    g_logger.info(f'Solver Execution Time Limit = {my_settings.r_execution_time_limit // 60 // 60:02}:'
                  f'{(my_settings.r_execution_time_limit // 60) % 60:02}:'
                  f'{my_settings.r_execution_time_limit % 60:02}')

    for solver_model_numbers_list in args.solver_models:
        for solver_model_numbers in solver_model_numbers_list:
            splitted_txt = solver_model_numbers.split()
            solver_name, model_numbers = splitted_txt[0], splitted_txt[1:]
            for i in model_numbers:
                my_settings.solver_model_combinations.append((
                    solver_name, AutoExecutorSettings.AVAILABLE_MODELS[int(i)]
                ))
    g_logger.info(f'Solver Model Combinations = {my_settings.solver_model_combinations}')

    my_settings.set_cpu_cores_per_solver(args.threads_per_solver_instance)
    g_logger.info(f'r_cpu_cores_per_solver = {my_settings.r_cpu_cores_per_solver}')

    if args.jobs == 0:
        my_settings.r_max_parallel_solvers = len(my_settings.solver_model_combinations)
    elif args.jobs == -1:
        my_settings.r_max_parallel_solvers = run_command_get_output('nproc')
    else:
        my_settings.r_max_parallel_solvers = args.jobs
    g_logger.info(f'r_max_parallel_solvers = {my_settings.r_max_parallel_solvers}')
    if my_settings.r_max_parallel_solvers < len(my_settings.solver_model_combinations):
        # TODO: Add more clear warning message explaining the technique used to get the results
        #       Result = Return the best result found in `EXECUTION_TIME_LIMIT` time among all solver model combinations.
        #                If no result is found, then wait until the first result is found and then return it.
        g_logger.warning('There is a possibility of more time being spent on execution'
                         'as all solver model combinations will not be running in parallel.'
                         f'\nSolver Model Combinations = {len(my_settings.solver_model_combinations)}')
    return my_settings


# REFER: https://stackoverflow.com/questions/1265665/how-can-i-check-if-a-string-represents-an-int-without-using-try-except
def parser_check_solver_models(val: str) -> str:
    """
    Validate that `val` is of the format 'SolverName ModelId [ModelId ...]', and:
        1. SolverName is present in `AutoExecutorSettings.AVAILABLE_SOLVERS`
        2. ModelId is an `int`
        3. ModelId is present in `AutoExecutorSettings.AVAILABLE_MODELS.keys()`

    Args:
        val: the value passed as commandline parameter

    Returns:
        `val` as it is
    """
    # Perform splitting based on spaces
    val_splitted = val.split()
    if len(val_splitted) == 0:
        raise argparse.ArgumentTypeError(f"no value passed")
    if len(val_splitted) == 1:
        if val_splitted[0] in AutoExecutorSettings.AVAILABLE_SOLVERS:
            raise argparse.ArgumentTypeError(f"no model numbers given")
        raise argparse.ArgumentTypeError(f"invalid solver name, "
                                         f"valid solvers = {AutoExecutorSettings.AVAILABLE_SOLVERS}")
    if val_splitted[0] not in AutoExecutorSettings.AVAILABLE_SOLVERS:
        raise argparse.ArgumentTypeError(f"invalid solver name, "
                                         f"valid solvers = {AutoExecutorSettings.AVAILABLE_SOLVERS}")
    for i in val_splitted[1:]:
        if not i.isdigit():
            raise argparse.ArgumentTypeError(f"model number should be int ('{i}' is not an int), valid model numbers = "
                                             f"{list(AutoExecutorSettings.AVAILABLE_MODELS.keys())}")
        if int(i) not in AutoExecutorSettings.AVAILABLE_MODELS.keys():
            raise argparse.ArgumentTypeError(f"invalid model number value: '{i}', valid model numbers = "
                                             f"{list(AutoExecutorSettings.AVAILABLE_MODELS.keys())}")
    return val


def parser_check_time_range(val: str) -> int:
    """
    Validate that `val` is of the format 'hh:mm:ss', and:
        1. hh can be converted to an `int`
        2. mm can be converted to an `int`
        3. ss can be converted to an `int`
        4. `0 <= hh`
        5. `0 <= mm < 60`
        6. `0 <= ss < 60`
        7. (hh*60*60 + mm*60 + ss) >= 30

    Args:
        val: the value passed as commandline parameter

    Returns:
        `val` converted to seconds
    """
    if val.count(':') != 2:
        raise argparse.ArgumentTypeError(f"invalid time value: '{val}', correct format is 'hh:mm:ss'")
    val_splitted = []
    # Handle inputs like '::30' --> ['0', '0', '30']
    for i in val.split(':'):
        val_splitted.append(i if len(i) > 0 else '0')
    # Ensure that each element is an integer
    for i in val_splitted:
        if not i.isdigit():
            raise argparse.ArgumentTypeError(f"invalid int value: '{val}'")
    # Check range of minutes
    if int(val_splitted[1]) >= 60:
        raise argparse.ArgumentTypeError(f"invalid minutes value: '{val}', 0 <= minutes < 60")
    # Check range of seconds
    if int(val_splitted[2]) >= 60:
        raise argparse.ArgumentTypeError(f"invalid seconds value: '{val}', 0 <= seconds < 60")
    # Convert 'hh:mm:ss' into seconds
    seconds = int(val_splitted[0]) * 60 * 60 + int(val_splitted[1]) * 60 + int(val_splitted[2])
    # Ensure that seconds >= 30
    if seconds < 30:
        raise argparse.ArgumentTypeError('minimum `N` is 30')
    return seconds


def parser_check_threads_int_range(c: str) -> int:
    """
    Validate that `c` can be converted to an `int`, and ensure that `int(c) >= 1`

    Args:
        c: the value passed as commandline parameter

    Returns:
        `c` converted to `int`
    """
    if not c.isdigit():
        raise argparse.ArgumentTypeError(f"invalid int value: '{c}'")
    val = int(c)
    if val < 1:
        raise argparse.ArgumentTypeError('minimum `N` is 1')
    return val


def parser_check_jobs_int_range(c: str) -> int:
    """
    Validate that `c` can be converted to an `int`, and ensure that `int(c) >= -1`

    Args:
        c: the value passed as commandline parameter

    Returns:
        `c` converted to `int`
    """
    if not c.isdigit():
        raise argparse.ArgumentTypeError(f"invalid int value: '{c}'")
    val = int(c)
    if val < -1:
        raise argparse.ArgumentTypeError('minimum `N` is -1')
    return val


def check_requirements():
    """
    Check if the basic requirements for this code/program/script to properly execute are satisfied or not
    """
    ok, res = run_command('which tmux')
    if not ok:
        print('`tmux` not installed')
        exit(1)
    ok, res = run_command('which bash')
    if not ok:
        print('`bash` not installed')
        exit(1)
    pass


if __name__ == '__main__':
    check_requirements()

    # Create the parser
    # REFER: https://realpython.com/command-line-interfaces-python-argparse/
    # REFER: https://stackoverflow.com/questions/19124304/what-does-metavar-and-action-mean-in-argparse-in-python
    # REFER: https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text
    # noinspection PyTypeChecker
    my_parser = argparse.ArgumentParser(
        prog='CalculateNetworkCost.py',
        description='Find cost of any graph/network (i.e. data/testcase file) '
                    'by executing various solvers using different models'
                    '\n\nNote: Baron output does not differentiate between valid and invalid solution.'
                    'Hence, there is a possibility that this program can exit but when solution vector'
                    'is being extracted, the program fails.',
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

    my_settings = update_settings(my_parser.parse_args())
    g_logger.info('START main program')
    try:
        main(my_settings)
    except Exception as e:
        g_logger.error(f'FIXME: {type(e)},\n\nException e:\n{e}\n\ntrace:\n{traceback.format_exc()}')
    g_logger.info('FINISHED main program')
