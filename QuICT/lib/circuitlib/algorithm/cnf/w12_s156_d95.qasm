OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
ccx q[10], q[11], q[8];
ccx q[9], q[8], q[10];
x q[2];
x q[5];
x q[9];
ccx q[6], q[11], q[9];
ccx q[2], q[5], q[11];
ccx q[6], q[11], q[9];
ccx q[2], q[5], q[11];
x q[2];
x q[5];
ccx q[9], q[8], q[10];
x q[1];
x q[4];
x q[8];
ccx q[7], q[11], q[8];
ccx q[1], q[4], q[11];
ccx q[7], q[11], q[8];
ccx q[1], q[4], q[11];
x q[1];
x q[4];
ccx q[9], q[8], q[10];
x q[2];
x q[5];
x q[9];
ccx q[6], q[11], q[9];
ccx q[2], q[5], q[11];
ccx q[6], q[11], q[9];
ccx q[2], q[5], q[11];
x q[2];
x q[5];
ccx q[9], q[8], q[10];
x q[1];
x q[4];
x q[8];
ccx q[7], q[11], q[8];
ccx q[1], q[4], q[11];
ccx q[7], q[11], q[8];
ccx q[1], q[4], q[11];
x q[1];
x q[4];
ccx q[10], q[11], q[8];
ccx q[9], q[8], q[11];
x q[4];
x q[9];
ccx q[5], q[11], q[9];
ccx q[1], q[4], q[11];
ccx q[5], q[11], q[9];
ccx q[1], q[4], q[11];
x q[4];
ccx q[9], q[8], q[11];
x q[5];
x q[7];
x q[8];
ccx q[7], q[11], q[8];
ccx q[2], q[5], q[11];
ccx q[7], q[11], q[8];
ccx q[2], q[5], q[11];
x q[5];
x q[7];
ccx q[9], q[8], q[11];
x q[4];
x q[9];
ccx q[5], q[11], q[9];
ccx q[1], q[4], q[11];
ccx q[5], q[11], q[9];
ccx q[1], q[4], q[11];
x q[4];
ccx q[9], q[8], q[11];
x q[5];
x q[7];
x q[8];
ccx q[7], q[11], q[8];
ccx q[2], q[5], q[11];
ccx q[7], q[11], q[8];
ccx q[2], q[5], q[11];
x q[5];
x q[7];
ccx q[10], q[11], q[8];
ccx q[9], q[8], q[10];
x q[2];
x q[5];
x q[9];
ccx q[6], q[11], q[9];
ccx q[2], q[5], q[11];
ccx q[6], q[11], q[9];
ccx q[2], q[5], q[11];
x q[2];
x q[5];
ccx q[9], q[8], q[10];
x q[1];
x q[4];
x q[8];
ccx q[7], q[11], q[8];
ccx q[1], q[4], q[11];
ccx q[7], q[11], q[8];
ccx q[1], q[4], q[11];
x q[1];
x q[4];
ccx q[9], q[8], q[10];
x q[2];
x q[5];
x q[9];
ccx q[6], q[11], q[9];
ccx q[2], q[5], q[11];
ccx q[6], q[11], q[9];
ccx q[2], q[5], q[11];
x q[2];
x q[5];
ccx q[9], q[8], q[10];
x q[1];
x q[4];
x q[8];
ccx q[7], q[11], q[8];
ccx q[1], q[4], q[11];
ccx q[7], q[11], q[8];
ccx q[1], q[4], q[11];
x q[1];
x q[4];
ccx q[10], q[11], q[8];
ccx q[9], q[8], q[11];
x q[4];
x q[9];
ccx q[5], q[11], q[9];
ccx q[1], q[4], q[11];
ccx q[5], q[11], q[9];
ccx q[1], q[4], q[11];
x q[4];
ccx q[9], q[8], q[11];
x q[5];
x q[7];
x q[8];
ccx q[7], q[11], q[8];
ccx q[2], q[5], q[11];
ccx q[7], q[11], q[8];
ccx q[2], q[5], q[11];
x q[5];
x q[7];
ccx q[9], q[8], q[11];
x q[4];
x q[9];
ccx q[5], q[11], q[9];
ccx q[1], q[4], q[11];
ccx q[5], q[11], q[9];
ccx q[1], q[4], q[11];
x q[4];
ccx q[9], q[8], q[11];
x q[5];
x q[7];
x q[8];
ccx q[7], q[11], q[8];
ccx q[2], q[5], q[11];
ccx q[7], q[11], q[8];
ccx q[2], q[5], q[11];
x q[5];
x q[7];
