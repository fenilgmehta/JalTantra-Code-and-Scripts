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

echo >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
echo "date    = '$(date)'" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
echo "whoami  = '$(whoami)'" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
echo "pwd     = '$(pwd)'" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
echo "\$0      = '${0}'" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
echo "\$1      = '${1}'" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
echo "\$2      = '${2}'" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
echo "python  = '$(which python)'" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
echo "python3 = '$(which python3)'" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
dirname "${0}" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
echo "$(dirname '$0')" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
echo "${FILE_PATH}" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log

if [[ -f "${1}" ]]; then
	# >>> conda initialize >>>
	# !! Contents within this block are managed by 'conda init' !!
	__conda_setup="$('/home/student/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
	if [ $? -eq 0 ]; then
	    eval "$__conda_setup"
	else
	    if [ -f "/home/student/miniconda3/etc/profile.d/conda.sh" ]; then
	        . "/home/student/miniconda3/etc/profile.d/conda.sh"
	    else
	        export PATH="/home/student/miniconda3/bin:$PATH"
	    fi  
	fi
	unset __conda_setup
	# <<< conda initialize <<<

	conda activate dev

	# REFER: https://stackoverflow.com/questions/876239/how-to-redirect-and-append-both-standard-output-and-standard-error-to-a-file-wit
	/home/student/miniconda3/envs/dev/bin/python3 CalculateNetworkCost.py -p "${1}" --solver-models 'octeract 1 2' --time "${2}" --debug > "${1}.log" 2>&1
else
	echo "ERROR: either file does not exist or it is not a file: '${1}'" >> log_jaltantra_CalculateNetworkCost_JaltantraLauncher.log
fi
