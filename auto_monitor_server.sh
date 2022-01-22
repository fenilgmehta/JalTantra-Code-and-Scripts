#!/usr/bin/bash
mkdir sys_stats
while true
do
	suffix=$(date +"%y-%m-%d_%H-%M-%S")
	# REFER: https://askubuntu.com/a/1168911
	echo | htop > "sys_stats/htop_${suffix}.out"
	sleep 10
done
