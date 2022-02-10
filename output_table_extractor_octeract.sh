#!/usr/bin/bash
echo -e "Processing file = '$1'"
cat "$1" | cut -c50- | awk -F'  +' '
BEGIN {f=0;}
/(mpiexec|The best solution|^[ \t]*$|^[^ ])/ {if(f==2) f+=1;}
{
    if(f==2) {
        # print;
        if (s[$1] == 0) {
            s[$1] = 1;
            gsub("s", "", $3);
            gsub("(^( |\t)+|( |\t)+$)", "", $1);
            gsub(" ", ",", $1);
            print $3 "," $1;
        }
    }
}
/---------------------/ {f+=1;}'
echo ','
