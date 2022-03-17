#!/bin/bash

# ls m1* | sort -V | xargs -I{} -n 1 ../../output_table_extractor_baron.sh {}
# ls m2* | sort -V | xargs -I{} -n 1 ../../output_table_extractor_baron.sh {}
# ls m3* | sort -V | xargs -I{} -n 1 ../../output_table_extractor_baron.sh {}
# ls m4* | sort -V | xargs -I{} -n 1 ../../output_table_extractor_baron.sh {}

echo -e "Processing file = '$1'"

OUTPUT_AWK=$(cat "$1" | awk -F'  +' '
BEGIN { f=0; rows=0; }
/^ *$/ { if(f==2) f+=1; }
{
    if(f==2) {
        # if (rows <= 3) {
        #     print $1 "--" $2 "--" $3 "--" $4 "--" $5 "--" $6;
        #     print;
        #     rows += 1;
        # }
        varTime = $4;
        varCost = $6;
        if (s[varCost] == 0) {
            s[varCost] = 1;
            print varTime "," varCost;
            rows += 1;
        }
    }
}
/  Iteration    Open nodes         Time \(s\)    Lower bound      Upper bound/ {f=2;}
END {
    for (; rows <= 7; rows += 1) {
        print ","
    }
}
')

# Perform compression of output values
# The first and last values are always printed, and the results between them are compressed
echo "${OUTPUT_AWK}" | python -c "
import sys
from math import ceil

# str.strip() is required as 'l' will have trailing new line character
lines = [l.strip() for l in sys.stdin]
lines_out = list()

if len(lines) <= 1+8:
    lines_out = lines
else:
    lines_out.append(lines[0] + ',Compressed')
    for i in range(1, len(lines) - 1, int(ceil((len(lines) - 2) / (8 - 2)))):
        lines_out.append(lines[i])
    lines_out.append(lines[-1])

for i in lines_out:
    print(i)
"

