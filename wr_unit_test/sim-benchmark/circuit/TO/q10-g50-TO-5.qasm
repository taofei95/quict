OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
cx q[8], q[5];
x q[0];
ccx q[7], q[3], q[4];
id q[6];
ccx q[9], q[2], q[1];
x q[6];
sdg q[0];
ccx q[4], q[7], q[6];
cx q[4], q[8];
cx q[5], q[0];
x q[0];
id q[6];
x q[2];
x q[4];
ccx q[0], q[4], q[7];
cx q[4], q[0];
x q[6];
s q[1];
ccx q[1], q[0], q[9];
x q[6];
ccx q[5], q[9], q[7];
ccx q[0], q[4], q[3];
x q[5];
x q[5];
cx q[7], q[2];
ccx q[9], q[7], q[8];
x q[5];
ccx q[8], q[2], q[3];
cx q[8], q[6];
x q[5];
cx q[8], q[0];
ccx q[5], q[3], q[6];
cx q[2], q[0];
x q[1];
x q[0];
x q[4];
cx q[6], q[9];
ccx q[3], q[7], q[0];
ccx q[2], q[4], q[6];
sdg q[3];
x q[1];
tdg q[6];
ccx q[4], q[5], q[9];
ccx q[4], q[6], q[2];
h q[6];
cx q[7], q[3];
cx q[5], q[6];
x q[9];
y q[0];
ccx q[4], q[6], q[2];