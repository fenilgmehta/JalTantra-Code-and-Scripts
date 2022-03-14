#!/usr/bin/bash
for var in "$@"
do
	# REFER: https://askubuntu.com/a/1168911
	head -c -10 "$var"  | tail -c +10
	echo "$var"
	sleep 0.5
done
