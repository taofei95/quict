OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
x q[1];
x q[2];
x q[5];
ccx q[2], q[7], q[5];
ccx q[0], q[1], q[7];
ccx q[2], q[7], q[5];
ccx q[0], q[1], q[7];
x q[1];
x q[2];
x q[6];
ccx q[2], q[7], q[6];
ccx q[0], q[1], q[7];
ccx q[2], q[7], q[6];
ccx q[0], q[1], q[7];
ccx q[5], q[6], q[4];
x q[1];
x q[2];
x q[5];
ccx q[2], q[7], q[5];
ccx q[0], q[1], q[7];
ccx q[2], q[7], q[5];
ccx q[0], q[1], q[7];
x q[1];
x q[2];
x q[6];
ccx q[2], q[7], q[6];
ccx q[0], q[1], q[7];
ccx q[2], q[7], q[6];
ccx q[0], q[1], q[7];
