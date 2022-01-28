# # REFER: https://docs.octeract.com/htg1001-how_to_use_ampl_with_octeract_engine
# # option solver "octeract-engine-4.0.0/bin/octeract-engine";
# # options octeract_options 'num_cores=48';
# option solver "/home/fenil/mtp/octeract_custom.sh";
# options octeract_options 'OUTPUT_FREQUENCY=1';
# options octeract_options 'OUTPUT_TIME_FREQUENCY=0.5';

# Octeract
option solver "octeract-engine-4.0.0/bin/octeract-engine";
# Gurobi
option solver "ampl_gurobi-engine/gurobi_ampl";

# ----------------------------------------------------------------------------------------------------

# --- 1 --- Data = d1_Sample_input_cycle_twoloop

# WORKING
model Files/Models/m1_basic.mod
data Files/Data/m1_m2/d1_Sample_input_cycle_twoloop.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# SOLVED to global optimality withing a minute or so
model Files/Models/m2_basic2.mod
data Files/Data/m1_m2/d1_Sample_input_cycle_twoloop.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# DONE previously
# q1, q2
model Files/Models/m3_descrete_segment.mod
data Files/Data/m3_m4/d1_Sample_input_cycle_twoloop.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# DONE
model Files/Models/m4_parallel_links.mod
data Files/Data/m3_m4/d1_Sample_input_cycle_twoloop.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# ----------------------------------------------------------------------------------------------------

# --- 2 --- Data = d2_Sample_input_cycle_hanoi

# WORKING
model Files/Models/m1_basic.mod
data Files/Data/m1_m2/d2_Sample_input_cycle_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# Not working due to demo licence is constrained to 300 variables
model Files/Models/m2_basic2.mod
data Files/Data/m1_m2/d2_Sample_input_cycle_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# DONE
model Files/Models/m3_descrete_segment.mod
data Files/Data/m3_m4/d2_Sample_input_cycle_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# SOLVED to global optimality withing a minute or so
model Files/Models/m4_parallel_links.mod
data Files/Data/m3_m4/d2_Sample_input_cycle_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# ----------------------------------------------------------------------------------------------------

# --- 3 --- Data = d3_Sample_input_double_hanoi

# Not working due to demo licence is constrained to 300 variables
model Files/Models/m1_basic.mod
data Files/Data/m1_m2/d3_Sample_input_double_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# Not working due to demo licence is constrained to 300 variables
model Files/Models/m2_basic2.mod
data Files/Data/m1_m2/d3_Sample_input_double_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# Not working due to demo licence is constrained to 300 variables
model Files/Models/m3_descrete_segment.mod
data Files/Data/m3_m4/d3_Sample_input_double_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# Not working due to demo licence is constrained to 300 variables
model Files/Models/m4_parallel_links.mod
data Files/Data/m3_m4/d3_Sample_input_double_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# ----------------------------------------------------------------------------------------------------

# --- 4 --- Data = d4_Sample_input_triple_hanoi

# Not working due to demo licence is constrained to 300 variables
model Files/Models/m1_basic.mod
data Files/Data/m1_m2/d4_Sample_input_triple_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# Not working due to demo licence is constrained to 300 variables
model Files/Models/m2_basic2.mod
data Files/Data/m1_m2/d4_Sample_input_triple_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# Not working due to demo licence is constrained to 300 variables
model Files/Models/m3_descrete_segment.mod
data Files/Data/m3_m4/d4_Sample_input_triple_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

# Not working due to demo licence is constrained to 300 variables
model Files/Models/m4_parallel_links.mod
data Files/Data/m3_m4/d4_Sample_input_triple_hanoi.dat
option solver "octeract-engine-4.0.0/bin/octeract-engine";
solve;

