OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
ccx q[5], q[4], q[0];
z q[5];
cx q[3], q[1];
cx q[3], q[4];
cx q[0], q[3];
x q[4];
ccx q[5], q[2], q[4];
x q[3];
id q[5];
ccx q[4], q[0], q[3];
ccx q[0], q[2], q[5];
cx q[0], q[4];
sdg q[0];
sdg q[5];
ccx q[4], q[5], q[2];
sdg q[1];
ccx q[3], q[5], q[2];
cx q[2], q[5];
x q[0];
x q[3];
ccx q[3], q[5], q[2];
h q[4];
x q[5];
ccx q[5], q[4], q[0];
x q[5];
ccx q[5], q[4], q[3];
x q[2];
ccx q[2], q[3], q[4];
y q[1];
cx q[1], q[4];