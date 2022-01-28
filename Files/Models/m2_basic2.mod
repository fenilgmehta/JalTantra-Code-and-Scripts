### SETS ###
set nodes;			### Set of nodes
set arcs within {i in nodes, j in nodes: i != j};
set pipes;			### Set of pipes available

### PARAMETERS ###
param d{pipes};  		### Diameter of each pipe
param L{arcs}; 		### Total length of each link
param E{nodes};  		### Elevation of all nodes including the source
param D{nodes};  			### Demand of each node except the source node
param C{pipes};   			### Cost per unit length for each commercially available pipe
param R{pipes}; 		### Roughness of available 
param P{nodes};  		### Minimum pressure required at each node
param Source;
param S{nodes,arcs};

### Undefined parameters ###
param q_M := -D[Source];				### Upper bound on flow variable
param q_m := 10**-20;					### Lower bound on flow variable
param omega := 10.68;			### SI Unit Constant for Hazen Williams Equation

### VARIABLES ###
var l{arcs,pipes} >= 0;		### Length of each pipe link
var q1{arcs}, >= q_m, <= q_M;	### Flow variable
var q2{arcs}, >= q_m, <= q_M;	### Flow variable
var h{nodes};

### OBJECTIVE ###
minimize total_cost : sum{(i,j) in arcs} sum{k in pipes}l[i,j,k]*C[k];		### Total cost as a sum of price per unit pipe * length of pipe

### Variable bounds ###
s.t. bound1{(i,j) in arcs, k in pipes}: l[i,j,k] <= L[i,j];
#s.t. bound2{(i,j) in arcs}: q1[i,j] >= 1000*(E[source]*( (R[1]^1.852) * (d[1]/1000)^4.87)/(omega*L[i,j]))^(1/1.852);
#s.t. bound3{(i,j) in arcs}: q2[i,j] >= 1000*(E[source]*( (R[1]^1.852) * (d[1]/1000)^4.87)/(omega*L[i,j]))^(1/1.852);


### CONSTRAINTS ###
s.t. con1{j in nodes}: sum{i in nodes : (i,j) in arcs}(q1[i,j]-q2[i,j]) = sum{i in nodes : (j,i) in arcs}(q1[j,i]-q2[j,i]) + D[j];

s.t. con2{i in nodes}: h[i] >= E[i] + P[i]; 

s.t. con3{(i,j) in arcs}: h[i] - h[j] = sum{k in pipes} omega * (((q1[i,j]*0.001))^1.852 - ((q2[i,j]*0.001))^1.852) * l[i,j,k] / ( (R[k]^1.852) * (d[k]/1000)^4.87);

s.t. con4{(i,j) in arcs}: sum{k in pipes} l[i,j,k] = L[i,j];

s.t. con5: h[Source] = E[Source];

s.t. con6{(i,j) in arcs}: q1[i,j]*q2[i,j]<=q_M*q_m;