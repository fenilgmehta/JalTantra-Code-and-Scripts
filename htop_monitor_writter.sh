#!/usr/bin/bash
mkdir -p sys_stats
while true
do
	suffix=$(date +"%y-%m-%d_%H-%M-%S")  # Example: 22-04-25_14-32-11
	# REFER: https://askubuntu.com/a/1168911
	echo | htop > "sys_stats/htop_${suffix}.out"
	sleep 10
done
