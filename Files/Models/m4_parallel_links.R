### SETS ###
set nodes;						### Set of nodes/vertexes
set cycle;						### Set of cycles
set links;						### Set of links
set pipes;						### Set of commercial pipes available

### PARAMETERS ###
param link_length{links};		### Total length of each arc/link
param elevation{nodes};			### Elevation of each node
param pressure{nodes};			### Minimum pressure required at each node
param demand{nodes};			### Demand of each node
param diameter{pipes};			### Diameter of each commercial pipe
param Cost{pipes};				### Cost per unit length of each commercial pipe
param Roughness{pipes};			### Roughness of each commercial pipe
param sourceHead;				### Head provided at the source (Same as source elevation in gravity fed system)
param F{nodes, links};			### Flow Direction Matrix
param S{nodes, links};			### Matrix for flow Direction in Spanning Tree
param C{cycle, links};			### Cycle Flow Direction Matrix

### Undefined parameters ###
param q_M := -demand['Node1'];	### Upper bound on flow variable (NOTE: Here, probably it is assumed that Node1 is the source)
								### `-D[Source]` is used because demand of source is `-1 * sum(demand of other nodes)`
param q_m := 10^(-30);			### Lower bound on flow variable (TODO: Probably, this needs to be set to 0 just like "m2_basic2_v2.R")
param omega := 10.68;			### SI Unit Constant for Hazen Williams Equation

### VARIABLES ###
var l{links, pipes}, >= 0;		### Length of each commercial pipe for each arc/link
var q1{links}, >= q_m,<=q_M;	### Flow variable
var q2{links}, >= q_m,<=q_M;	### Flow variable
var x1 binary;

### OBJECTIVE ###
minimize total_cost : sum{i in links, j in pipes}l[i,j]*Cost[j];	### Total cost as a sum of "length of the commercial pipe * cost per unit length of the commercial pipe"

### Variable bounds ###
s.t. bound1{i in links, j in pipes}: l[i,j] <= link_length[i];

### CONSTRAINTS ###
s.t. con1{i in nodes}: sum{j in links} F[i,j]*(q1[j] - q2[j]) = demand[i];

s.t. con2{i in cycle}: sum{j in links}sum{k in pipes} C[i,j]*omega*l[j,k]*((q1[j]/1000)^1.852 - (q2[j]/1000)^1.852)/((Roughness[k]^1.852)*((diameter[k]/1000)**4.87)) = 0;

s.t. con3a{i in nodes}: sum{j in links}sum{k in pipes} S[i,j]*omega*l[j,k]*((q1[j]/1000)^1.852 - (q2[j]/1000)^1.852)/((Roughness[k]^1.852)*((diameter[k]/1000)**4.87)) <= sourceHead - elevation[i] - pressure[i];
s.t. con3b{i in nodes}: sum{j in links}sum{k in pipes} S[i,j]*omega*l[j,k]*((q1[j]/1000)^1.852 - (q2[j]/1000)^1.852)/((Roughness[k]^1.852)*((diameter[k]/1000)**4.87)) >= 0;

s.t. con4{i in links}: q1[i]*q2[i] <=q_M*q_m;

s.t. con5{i in links}: sum{j in pipes} l[i,j] = link_length[i];
