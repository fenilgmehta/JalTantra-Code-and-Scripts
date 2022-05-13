### SETS ###
set nodes;			### Set of nodes/vertexes
set pipes;			### Set of commercial pipes available
set arcs within {i in nodes, j in nodes: i != j};	### Set of arcs/links/edges

### PARAMETERS ###
param L{arcs};		### Total length of each arc/link
param E{nodes};		### Elevation of each node
param P{nodes};		### Minimum pressure required at each node
param D{nodes};		### Demand of each node
param d{pipes};		### Diameter of each commercial pipe
param C{pipes};		### Cost per unit length of each commercial pipe
param R{pipes};		### Roughness of each commercial pipe
param Source;		### Source node ID

### Undefined parameters ###
param q_M := -D[Source];		### Upper bound on flow variable
								### `-D[Source]` is used because demand of source is `-1 * sum(demand of other nodes)`
param q_m := D[Source];			### Lower bound on flow variable
param omega := 10.68;			### SI Unit Constant for Hazen Williams Equation

### VARIABLES ###
var l{arcs,pipes} >= 0;			### Length of each commercial pipe for each arc/link
var q{arcs}, >= q_m, <= q_M;	### Flow variable
var h{nodes};					### Head

### OBJECTIVE ###
minimize total_cost : sum{(i,j) in arcs} sum{k in pipes}l[i,j,k]*C[k];	### Total cost as a sum of "length of the commercial pipe * cost per unit length of the commercial pipe"

### Variable bounds ###
s.t. bound1{(i,j) in arcs, k in pipes}: l[i,j,k] <= L[i,j];

### CONSTRAINTS ###
s.t. con1{j in nodes}: sum{i in nodes : (i,j) in arcs}q[i,j] = sum{i in nodes : (j,i) in arcs}q[j,i] + D[j];

s.t. con2{i in nodes}: h[i] >= E[i] + P[i];

s.t. con3{(i,j) in arcs}: h[i] - h[j] = (q[i,j] * abs(q[i,j])^0.852) * (0.001^1.852) * sum{k in pipes} omega * l[i,j,k] / ( (R[k]^1.852) * (d[k]/1000)^4.87);

s.t. con4{(i,j) in arcs}: sum{k in pipes} l[i,j,k] = L[i,j];

s.t. con5: h[Source] = E[Source];

#s.t. bound7{(i,j) in arcs}: abs(q[i,j]) >= 10**-;
