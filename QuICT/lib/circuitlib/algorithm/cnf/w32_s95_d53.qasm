OPENQASM 2.0;
include "qelib1.inc";
qreg q[32];
creg c[32];
ccx q[29], q[28], q[30];
x q[25];
x q[29];
ccx q[25], q[31], q[29];
ccx q[4], q[19], q[31];
ccx q[25], q[31], q[29];
ccx q[4], q[19], q[31];
x q[25];
ccx q[29], q[28], q[30];
x q[10];
x q[20];
x q[24];
x q[28];
ccx q[24], q[31], q[28];
ccx q[10], q[20], q[31];
ccx q[24], q[31], q[28];
ccx q[10], q[20], q[31];
x q[10];
x q[20];
x q[24];
ccx q[29], q[28], q[30];
x q[25];
x q[29];
ccx q[25], q[31], q[29];
ccx q[4], q[19], q[31];
ccx q[25], q[31], q[29];
ccx q[4], q[19], q[31];
x q[25];
ccx q[29], q[28], q[30];
x q[10];
x q[20];
x q[24];
x q[28];
ccx q[24], q[31], q[28];
ccx q[10], q[20], q[31];
ccx q[24], q[31], q[28];
ccx q[10], q[20], q[31];
x q[10];
x q[20];
x q[24];
x q[22];
x q[31];
ccx q[22], q[30], q[31];
ccx q[0], q[3], q[30];
ccx q[22], q[30], q[31];
ccx q[0], q[3], q[30];
x q[22];
ccx q[30], q[31], q[28];
ccx q[29], q[28], q[30];
x q[25];
x q[29];
ccx q[25], q[31], q[29];
ccx q[4], q[19], q[31];
ccx q[25], q[31], q[29];
ccx q[4], q[19], q[31];
x q[25];
ccx q[29], q[28], q[30];
x q[10];
x q[20];
x q[24];
x q[28];
ccx q[24], q[31], q[28];
ccx q[10], q[20], q[31];
ccx q[24], q[31], q[28];
ccx q[10], q[20], q[31];
x q[10];
x q[20];
x q[24];
ccx q[29], q[28], q[30];
x q[25];
x q[29];
ccx q[25], q[31], q[29];
ccx q[4], q[19], q[31];
ccx q[25], q[31], q[29];
ccx q[4], q[19], q[31];
x q[25];
ccx q[29], q[28], q[30];
x q[10];
x q[20];
x q[24];
x q[28];
ccx q[24], q[31], q[28];
ccx q[10], q[20], q[31];
ccx q[24], q[31], q[28];
ccx q[10], q[20], q[31];
x q[10];
x q[20];
x q[24];
x q[22];
x q[31];
ccx q[22], q[30], q[31];
ccx q[0], q[3], q[30];
ccx q[22], q[30], q[31];
ccx q[0], q[3], q[30];
x q[22];
