#!/bin/bash

# Usage: bash CalculateNetworkCost_JaltantraLauncher.sh "/path/to/network_file" "0:5:0"
#             $0                                        $1                      $2

# Change the working dir to the parent of `CalculateNetworkCost.py`
# REFER: https://stackoverflow.com/questions/630372/determine-the-path-of-the-executing-bash-script
# NOTE: cd "$(dirname '$0')"    did not work
# NOTE: cd "$(dirname \"$0\")"  did not work
# NOTE: cd "$(dirname "$0")"    did worked
FILE_PATH=$(dirname "$0")
cd "${FILE_PATH}"

LOG_FILE='log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log'
MINICONDA_HOME='/home/student/miniconda3'

echo >> "${LOG_FILE}"
echo "date    = '$(date)'" >> "${LOG_FILE}"
echo "whoami  = '$(whoami)'" >> "${LOG_FILE}"
echo "pwd     = '$(pwd)'" >> "${LOG_FILE}"
echo "\$0      = '${0}'" >> "${LOG_FILE}"
echo "\$1      = '${1}'" >> "${LOG_FILE}"
echo "\$2      = '${2}'" >> "${LOG_FILE}"
echo "python  = '$(which python)'" >> "${LOG_FILE}"
echo "python3 = '$(which python3)'" >> "${LOG_FILE}"
dirname "${0}" >> "${LOG_FILE}"
echo "$(dirname '$0')" >> "${LOG_FILE}"
echo "${FILE_PATH}" >> "${LOG_FILE}"

if [[ -f "${1}" ]]; then
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$(\"${MINICONDA_HOME}/bin/conda\" 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "${MINICONDA_HOME}/etc/profile.d/conda.sh" ]; then
            . "${MINICONDA_HOME}/etc/profile.d/conda.sh"
        else
            export PATH="${MINICONDA_HOME}/bin:$PATH"
        fi  
    fi
    unset __conda_setup
    # <<< conda initialize <<<

    conda activate dev

    # REFER: https://stackoverflow.com/questions/876239/how-to-redirect-and-append-both-standard-output-and-standard-error-to-a-file-wit
    "${MINICONDA_HOME}/envs/dev/bin/python3" CalculateNetworkCost.py -p "${1}" --solver-models 'octeract 1 2' --time "${2}" --debug > "${1}.log" 2>&1
else
    echo "ERROR: either file does not exist or it is not a file: '${1}'" >> "${LOG_FILE}"
fi
