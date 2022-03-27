### SETS ###
set nodes;						### Set of nodes
set cycle;						### Set of cycles
set links;						### Set of links
set pipes;						### Set of pipes available

### PARAMETERS ###
param diameter{pipes};			### Diameter of each pipe
param link_length{links};		### Total length of each link
param elevation{nodes};			### Elevation of all nodes including the source
param demand{nodes};			### Demand of each node except the source node
param Cost{pipes};				### Cost per unit length for each commercially available pipe
param Roughness{pipes};			### Roughness of available
param sourceHead;				### Head provided at the source (Same as source elevation in gravity fed system)
param pressure{nodes};			### Minimum pressure required at each node
param F{nodes, links};			### Flow Direction Matrix
param S{nodes, links};			### Matrix for flow Direction in Spanning Tree
param C{cycle, links};			### Cycle Flow Direction Matrix

### Undefined parameters ###
param q_M := -demand['Node1'];	### Upper bound on flow variable
param q_m := 10**-1;			### Lower bound on flow variable
param omega := 10.68;			### SI Unit Constant for Hazen Williams Equation

### VARIABLES ###
var l{links, pipes} >= 0;		### Length of each pipe link
var q{links}, >= -q_M, <= q_M;	### Flow variable

### OBJECTIVE ###
minimize total_cost : sum{i in links, j in pipes}l[i,j]*Cost[j];	### Total cost as a sum of price per unit pipe * length of pipe

### Variable bounds ###
s.t. bound1{i in links, j in pipes}: l[i,j] <= link_length[i];

### CONSTRAINTS ###
s.t. con1{i in nodes}: sum{j in links} F[i,j]*q[j] = demand[i];

s.t. con2{i in cycle}: sum{j in links}sum{k in pipes} C[i,j]*omega*l[j,k]*(q[j]*0.001)*(abs((q[j]*0.001))**0.852)/((Roughness[k]**1.852)*((diameter[k]/1000)**4.87)) = 0;

s.t. con3a{i in nodes}: sum{j in links}sum{k in pipes} S[i,j]*omega*l[j,k]*(q[j]*0.001)*(abs((q[j]*0.001))^0.852)/((Roughness[k]**1.852)*((diameter[k]/1000)^4.87)) <= sourceHead - elevation[i] - pressure[i];
s.t. con3b{i in nodes}: sum{j in links}sum{k in pipes} S[i,j]*omega*l[j,k]*(q[j]*0.001)*(abs((q[j]*0.001))^0.852)/((Roughness[k]**1.852)*((diameter[k]/1000)^4.87)) >= 0;

s.t. con4{i in links}: q_m <= abs(q[i]);
s.t. con5{i in links}: sum{j in pipes} l[i,j] = link_length[i];
