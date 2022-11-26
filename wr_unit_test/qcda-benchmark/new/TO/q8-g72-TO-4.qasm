OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
z q[4];
ccx q[0], q[3], q[1];
ccx q[6], q[7], q[3];
z q[2];
ccx q[0], q[1], q[4];
cx q[2], q[1];
x q[6];
tdg q[3];
h q[3];
ccx q[6], q[2], q[3];
y q[3];
cx q[6], q[5];
ccx q[1], q[5], q[3];
t q[3];
ccx q[6], q[1], q[2];
ccx q[5], q[2], q[7];
x q[7];
sdg q[2];
ccx q[7], q[0], q[2];
tdg q[6];
x q[6];
ccx q[4], q[2], q[0];
cx q[2], q[4];
x q[7];
ccx q[7], q[1], q[2];
x q[5];
cx q[4], q[7];
cx q[0], q[5];
x q[1];
cx q[2], q[7];
y q[0];
cx q[4], q[7];
x q[4];
ccx q[3], q[7], q[4];
ccx q[3], q[2], q[0];
ccx q[7], q[3], q[1];
x q[4];
cx q[4], q[2];
ccx q[4], q[2], q[0];
id q[1];
x q[4];
cx q[4], q[1];
s q[4];
s q[6];
h q[2];
cx q[4], q[7];
x q[1];
ccx q[2], q[6], q[0];
cx q[6], q[1];
cx q[7], q[1];
cx q[5], q[2];
x q[1];
cx q[5], q[0];
ccx q[4], q[1], q[5];
x q[5];
cx q[1], q[2];
ccx q[2], q[6], q[5];
id q[1];
cx q[6], q[3];
x q[2];
z q[2];
y q[1];
y q[2];
y q[7];
cx q[0], q[1];
cx q[3], q[2];
x q[4];
t q[6];
ccx q[4], q[0], q[7];
z q[4];
x q[5];
cx q[1], q[7];