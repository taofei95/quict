OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
s q[5];
ccx q[5], q[0], q[1];
cx q[3], q[0];
x q[3];
ccx q[0], q[3], q[2];
sdg q[1];
x q[6];
t q[5];
cx q[4], q[0];
ccx q[5], q[2], q[3];
sdg q[6];
ccx q[0], q[4], q[3];
cx q[4], q[2];
ccx q[6], q[0], q[4];
x q[4];
cx q[6], q[0];
cx q[0], q[3];
x q[0];
ccx q[5], q[3], q[0];
y q[6];
t q[6];
cx q[0], q[6];
x q[3];
ccx q[3], q[4], q[6];
x q[1];
cx q[4], q[6];
x q[3];
ccx q[0], q[3], q[5];
cx q[6], q[0];
s q[1];
cx q[1], q[4];
x q[0];
ccx q[1], q[5], q[0];
ccx q[0], q[3], q[4];
cx q[6], q[1];
ccx q[5], q[3], q[0];
cx q[6], q[1];
x q[2];
x q[4];
cx q[1], q[2];
ccx q[1], q[6], q[2];
cx q[0], q[4];
x q[1];
x q[0];
cx q[0], q[3];
id q[3];
ccx q[1], q[3], q[6];
z q[6];
ccx q[1], q[3], q[4];
cx q[2], q[3];
cx q[1], q[2];
cx q[4], q[1];
tdg q[4];
cx q[3], q[4];
x q[4];
ccx q[5], q[4], q[6];
x q[1];
sdg q[6];
t q[2];
cx q[5], q[6];
s q[5];
tdg q[4];
x q[0];
ccx q[0], q[2], q[1];
cx q[4], q[3];
s q[5];
ccx q[2], q[4], q[3];
cx q[5], q[4];
ccx q[5], q[2], q[0];
x q[1];
cx q[0], q[1];
ccx q[0], q[1], q[4];
x q[4];
ccx q[3], q[0], q[5];
x q[1];
h q[1];
cx q[6], q[3];
ccx q[4], q[2], q[6];
t q[3];
cx q[1], q[3];
z q[2];
x q[3];
s q[0];
t q[4];
id q[1];
ccx q[1], q[4], q[5];
ccx q[4], q[3], q[0];
cx q[1], q[4];
h q[1];
x q[0];
cx q[4], q[6];