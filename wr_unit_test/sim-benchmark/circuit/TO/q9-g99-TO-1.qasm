OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
ccx q[0], q[7], q[6];
ccx q[7], q[1], q[8];
x q[1];
ccx q[4], q[2], q[0];
x q[0];
x q[3];
ccx q[1], q[0], q[5];
x q[2];
t q[5];
cx q[4], q[5];
cx q[7], q[3];
sdg q[5];
ccx q[8], q[2], q[0];
x q[3];
x q[7];
ccx q[2], q[7], q[3];
x q[2];
tdg q[2];
ccx q[2], q[7], q[0];
ccx q[2], q[7], q[8];
x q[6];
cx q[1], q[0];
ccx q[2], q[4], q[6];
cx q[8], q[1];
y q[6];
x q[5];
sdg q[2];
cx q[2], q[6];
ccx q[6], q[8], q[0];
id q[0];
y q[7];
cx q[8], q[5];
cx q[2], q[1];
cx q[4], q[7];
cx q[5], q[0];
id q[1];
sdg q[7];
ccx q[4], q[7], q[6];
x q[0];
ccx q[3], q[6], q[8];
ccx q[6], q[1], q[7];
cx q[2], q[4];
sdg q[3];
ccx q[8], q[5], q[6];
x q[7];
cx q[5], q[1];
cx q[0], q[8];
x q[2];
cx q[8], q[0];
cx q[0], q[6];
ccx q[4], q[5], q[1];
ccx q[3], q[5], q[2];
t q[8];
ccx q[5], q[2], q[3];
cx q[7], q[0];
ccx q[1], q[8], q[4];
z q[5];
cx q[3], q[1];
ccx q[2], q[1], q[3];
cx q[1], q[3];
h q[6];
y q[8];
ccx q[4], q[5], q[1];
x q[2];
ccx q[1], q[7], q[4];
x q[4];
x q[5];
ccx q[1], q[6], q[0];
sdg q[7];
ccx q[5], q[8], q[2];
x q[6];
s q[5];
y q[7];
x q[7];
cx q[0], q[3];
t q[7];
x q[4];
sdg q[5];
t q[1];
x q[0];
cx q[6], q[8];
sdg q[0];
sdg q[2];
cx q[7], q[1];
cx q[5], q[6];
z q[8];
cx q[7], q[2];
h q[2];
cx q[7], q[5];
x q[3];
cx q[1], q[3];
x q[2];
x q[2];
cx q[5], q[3];
x q[2];
cx q[6], q[4];
sdg q[2];
cx q[4], q[2];
sdg q[4];