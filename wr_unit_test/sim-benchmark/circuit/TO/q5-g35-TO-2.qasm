OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
sdg q[2];
ccx q[1], q[3], q[0];
h q[0];
ccx q[4], q[3], q[2];
x q[3];
sdg q[2];
cx q[4], q[0];
ccx q[3], q[0], q[1];
cx q[1], q[4];
t q[1];
id q[0];
ccx q[3], q[0], q[2];
t q[4];
x q[4];
x q[1];
cx q[3], q[4];
ccx q[2], q[0], q[1];
ccx q[3], q[2], q[1];
y q[0];
cx q[3], q[2];
cx q[2], q[4];
ccx q[3], q[1], q[2];
ccx q[2], q[1], q[4];
ccx q[4], q[3], q[0];
cx q[0], q[1];
ccx q[1], q[0], q[2];
x q[4];
cx q[0], q[4];
x q[1];
s q[4];
x q[4];
t q[1];
tdg q[4];
x q[0];
id q[4];