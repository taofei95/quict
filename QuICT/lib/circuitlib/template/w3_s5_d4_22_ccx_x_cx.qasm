OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
ccx q[0], q[1], q[2];
x q[1];
ccx q[0], q[1], q[2];
x q[1];
cx q[0], q[2];
