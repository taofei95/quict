OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
ccx q[8], q[9], q[6];
ccx q[7], q[6], q[8];
x q[1];
x q[3];
x q[7];
ccx q[4], q[9], q[7];
ccx q[1], q[3], q[9];
ccx q[4], q[9], q[7];
ccx q[1], q[3], q[9];
x q[1];
x q[3];
ccx q[7], q[6], q[8];
x q[2];
x q[5];
x q[6];
ccx q[5], q[9], q[6];
ccx q[1], q[2], q[9];
ccx q[5], q[9], q[6];
ccx q[1], q[2], q[9];
x q[2];
x q[5];
ccx q[7], q[6], q[8];
x q[1];
x q[3];
x q[7];
ccx q[4], q[9], q[7];
ccx q[1], q[3], q[9];
ccx q[4], q[9], q[7];
ccx q[1], q[3], q[9];
x q[1];
x q[3];
ccx q[7], q[6], q[8];
x q[2];
x q[5];
x q[6];
ccx q[5], q[9], q[6];
ccx q[1], q[2], q[9];
ccx q[5], q[9], q[6];
ccx q[1], q[2], q[9];
x q[2];
x q[5];
ccx q[8], q[9], q[6];
x q[1];
x q[2];
x q[9];
ccx q[3], q[8], q[9];
ccx q[1], q[2], q[8];
ccx q[3], q[8], q[9];
ccx q[1], q[2], q[8];
x q[1];
x q[2];
ccx q[8], q[9], q[6];
ccx q[7], q[6], q[8];
x q[1];
x q[3];
x q[7];
ccx q[4], q[9], q[7];
ccx q[1], q[3], q[9];
ccx q[4], q[9], q[7];
ccx q[1], q[3], q[9];
x q[1];
x q[3];
ccx q[7], q[6], q[8];
x q[2];
x q[5];
x q[6];
ccx q[5], q[9], q[6];
ccx q[1], q[2], q[9];
ccx q[5], q[9], q[6];
ccx q[1], q[2], q[9];
x q[2];
x q[5];
ccx q[7], q[6], q[8];
x q[1];
x q[3];
x q[7];
ccx q[4], q[9], q[7];
ccx q[1], q[3], q[9];
ccx q[4], q[9], q[7];
ccx q[1], q[3], q[9];
x q[1];
x q[3];
ccx q[7], q[6], q[8];
x q[2];
x q[5];
x q[6];
ccx q[5], q[9], q[6];
ccx q[1], q[2], q[9];
ccx q[5], q[9], q[6];
ccx q[1], q[2], q[9];
x q[2];
x q[5];
ccx q[8], q[9], q[6];
x q[1];
x q[2];
x q[9];
ccx q[3], q[8], q[9];
ccx q[1], q[2], q[8];
ccx q[3], q[8], q[9];
ccx q[1], q[2], q[8];
x q[1];
x q[2];
