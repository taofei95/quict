OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
x q[3];
x q[4];
ccx q[2], q[3], q[4];
x q[3];
