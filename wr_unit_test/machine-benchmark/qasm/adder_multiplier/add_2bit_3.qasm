OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
ccx q[0], q[3], q[5];
h q[2];
crz(pi/2) q[1], q[2];
h q[1];
crz(pi) q[4], q[1];
crz(pi/2) q[4], q[2];
crz(pi) q[5], q[1];
crz(pi/2) q[5], q[2];
h q[1];
crz(-pi/2) q[1], q[2];
h q[2];
ccx q[0], q[3], q[5];
cx q[3], q[0];
