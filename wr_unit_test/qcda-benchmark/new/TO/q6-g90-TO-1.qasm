OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
tdg q[4];
cx q[5], q[0];
id q[5];
ccx q[5], q[2], q[1];
ccx q[5], q[3], q[0];
x q[3];
x q[1];
x q[2];
x q[1];
cx q[0], q[2];
ccx q[5], q[4], q[3];
x q[1];
x q[2];
cx q[0], q[3];
x q[1];
cx q[0], q[4];
x q[2];
ccx q[2], q[5], q[3];
h q[4];
s q[0];
ccx q[2], q[3], q[1];
tdg q[1];
t q[5];
cx q[2], q[1];
sdg q[0];
x q[4];
sdg q[0];
ccx q[0], q[1], q[2];
x q[4];
x q[3];
cx q[3], q[0];
cx q[0], q[5];
x q[0];
ccx q[1], q[0], q[4];
id q[1];
ccx q[2], q[0], q[5];
tdg q[4];
id q[5];
ccx q[1], q[3], q[5];
y q[5];
y q[2];
x q[4];
x q[3];
ccx q[2], q[1], q[0];
cx q[2], q[5];
cx q[2], q[3];
x q[0];
x q[5];
t q[2];
ccx q[5], q[1], q[2];
x q[0];
x q[2];
ccx q[4], q[2], q[3];
x q[1];
cx q[3], q[0];
ccx q[3], q[5], q[0];
x q[4];
t q[2];
x q[3];
x q[4];
x q[1];
ccx q[0], q[3], q[2];
cx q[0], q[5];
cx q[2], q[3];
cx q[5], q[1];
id q[5];
x q[4];
cx q[3], q[2];
tdg q[3];
cx q[3], q[0];
t q[4];
cx q[2], q[0];
t q[5];
ccx q[0], q[5], q[1];
cx q[4], q[3];
cx q[1], q[0];
x q[2];
cx q[3], q[1];
ccx q[5], q[0], q[2];
z q[2];
tdg q[0];
cx q[0], q[4];
ccx q[5], q[3], q[0];
ccx q[1], q[4], q[2];
z q[3];
ccx q[2], q[0], q[3];
h q[4];
ccx q[4], q[5], q[2];
x q[3];
y q[2];