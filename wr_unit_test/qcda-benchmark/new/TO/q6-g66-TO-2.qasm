OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
ccx q[2], q[4], q[5];
ccx q[4], q[1], q[2];
x q[5];
y q[4];
x q[5];
x q[0];
y q[5];
cx q[4], q[2];
cx q[5], q[0];
x q[4];
ccx q[3], q[0], q[5];
sdg q[2];
x q[3];
cx q[5], q[3];
cx q[2], q[1];
cx q[1], q[0];
ccx q[2], q[1], q[0];
z q[0];
tdg q[1];
ccx q[3], q[5], q[2];
h q[5];
cx q[3], q[1];
cx q[5], q[0];
x q[0];
cx q[3], q[1];
x q[1];
ccx q[4], q[2], q[5];
ccx q[2], q[4], q[5];
cx q[2], q[0];
ccx q[5], q[4], q[3];
ccx q[4], q[0], q[2];
cx q[2], q[1];
s q[0];
x q[1];
x q[2];
ccx q[2], q[3], q[1];
x q[3];
cx q[1], q[3];
x q[0];
ccx q[5], q[3], q[1];
cx q[0], q[1];
tdg q[0];
s q[2];
x q[4];
x q[3];
ccx q[1], q[4], q[2];
sdg q[3];
z q[2];
cx q[5], q[3];
cx q[1], q[3];
ccx q[3], q[0], q[4];
x q[4];
tdg q[3];
cx q[0], q[4];
h q[4];
x q[1];
ccx q[4], q[2], q[0];
t q[1];
sdg q[5];
cx q[5], q[0];
x q[5];
ccx q[2], q[4], q[0];
y q[1];
cx q[0], q[2];
s q[0];
ccx q[4], q[0], q[2];