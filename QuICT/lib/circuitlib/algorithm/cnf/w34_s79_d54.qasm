OPENQASM 2.0;
include "qelib1.inc";
qreg q[34];
creg c[34];
ccx q[31], q[30], q[32];
x q[31];
ccx q[23], q[33], q[31];
ccx q[1], q[3], q[33];
ccx q[23], q[33], q[31];
ccx q[1], q[3], q[33];
ccx q[31], q[30], q[32];
x q[7];
x q[19];
x q[30];
ccx q[19], q[33], q[30];
ccx q[7], q[16], q[33];
ccx q[19], q[33], q[30];
ccx q[7], q[16], q[33];
x q[7];
x q[19];
ccx q[31], q[30], q[32];
x q[31];
ccx q[23], q[33], q[31];
ccx q[1], q[3], q[33];
ccx q[23], q[33], q[31];
ccx q[1], q[3], q[33];
ccx q[31], q[30], q[32];
x q[7];
x q[19];
x q[30];
ccx q[19], q[33], q[30];
ccx q[7], q[16], q[33];
ccx q[19], q[33], q[30];
ccx q[7], q[16], q[33];
x q[7];
x q[19];
x q[18];
x q[33];
ccx q[25], q[32], q[33];
ccx q[2], q[18], q[32];
ccx q[25], q[32], q[33];
ccx q[2], q[18], q[32];
x q[18];
ccx q[32], q[33], q[30];
ccx q[31], q[30], q[32];
x q[31];
ccx q[23], q[33], q[31];
ccx q[1], q[3], q[33];
ccx q[23], q[33], q[31];
ccx q[1], q[3], q[33];
ccx q[31], q[30], q[32];
x q[7];
x q[19];
x q[30];
ccx q[19], q[33], q[30];
ccx q[7], q[16], q[33];
ccx q[19], q[33], q[30];
ccx q[7], q[16], q[33];
x q[7];
x q[19];
ccx q[31], q[30], q[32];
x q[31];
ccx q[23], q[33], q[31];
ccx q[1], q[3], q[33];
ccx q[23], q[33], q[31];
ccx q[1], q[3], q[33];
ccx q[31], q[30], q[32];
x q[7];
x q[19];
x q[30];
ccx q[19], q[33], q[30];
ccx q[7], q[16], q[33];
ccx q[19], q[33], q[30];
ccx q[7], q[16], q[33];
x q[7];
x q[19];
x q[18];
x q[33];
ccx q[25], q[32], q[33];
ccx q[2], q[18], q[32];
ccx q[25], q[32], q[33];
ccx q[2], q[18], q[32];
x q[18];
