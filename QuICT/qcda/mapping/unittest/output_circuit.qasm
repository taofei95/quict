OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
cx q[5], q[4];
tdg q[4];
t q[5];
cx q[3], q[1];
x q[3];
cx q[3], q[5];
t q[5];
swap q[4], q[5];
cx q[3], q[5];
tdg q[3];
cx q[4], q[5];
h q[4];
swap q[5], q[6];
cx q[4], q[5];
x q[4];
t q[4];
t q[4];
swap q[1], q[3];
cx q[5], q[3];
tdg q[5];
cx q[3], q[5];
cx q[4], q[5];
h q[4];
