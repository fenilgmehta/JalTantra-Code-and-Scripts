#!/usr/bin/bash

# ls m1* | sort -V | xargs -I{} -n 1 ../../output_table_extractor_octeract.sh {}
# ls m2* | sort -V | xargs -I{} -n 1 ../../output_table_extractor_octeract.sh {}
# ls m3* | sort -V | xargs -I{} -n 1 ../../output_table_extractor_octeract.sh {}
# ls m4* | sort -V | xargs -I{} -n 1 ../../output_table_extractor_octeract.sh {}

echo -e "Processing file = '$1'"

cat "$1" | cut -c50- | awk -F'  +' '
BEGIN {f=0; rows=0}
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
            rows += 1;
        }
    }
}
/---------------------/ {f+=1;}
END {
    for (; rows < 8; rows += 1) {
        print ","
    }
}
'
