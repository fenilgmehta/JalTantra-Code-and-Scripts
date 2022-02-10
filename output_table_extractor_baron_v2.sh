#!/bin/bash

# NOTE: This script is for verifying the original "output_table_extractor_baron.sh" program
echo -e "Processing file = '$1'"
cat "$1" | cut -c64- | awk '
BEGIN { f=0; rows=0; }
/^ *$/ { if(f==2) f+=1; }
{
    if (f==2)
        print;
}
/Upper bound/ {f=2;}
' | uniq
