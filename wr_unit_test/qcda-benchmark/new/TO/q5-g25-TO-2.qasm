OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
cx q[2], q[0];
ccx q[4], q[1], q[2];
x q[3];
id q[2];
sdg q[0];
x q[2];
cx q[0], q[3];
cx q[2], q[3];
h q[3];
tdg q[0];
h q[3];
ccx q[0], q[2], q[4];
id q[3];
x q[4];
cx q[1], q[3];
z q[1];
cx q[3], q[2];
x q[3];
x q[4];
x q[3];
cx q[3], q[1];
ccx q[4], q[1], q[3];
ccx q[0], q[2], q[1];
tdg q[1];
h q[0];