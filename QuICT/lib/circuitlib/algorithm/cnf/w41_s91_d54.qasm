OPENQASM 2.0;
include "qelib1.inc";
qreg q[41];
creg c[41];
ccx q[38], q[37], q[39];
x q[3];
x q[32];
x q[38];
ccx q[32], q[40], q[38];
ccx q[3], q[20], q[40];
ccx q[32], q[40], q[38];
ccx q[3], q[20], q[40];
x q[3];
x q[32];
ccx q[38], q[37], q[39];
x q[3];
x q[37];
ccx q[20], q[40], q[37];
ccx q[3], q[4], q[40];
ccx q[20], q[40], q[37];
ccx q[3], q[4], q[40];
x q[3];
ccx q[38], q[37], q[39];
x q[3];
x q[32];
x q[38];
ccx q[32], q[40], q[38];
ccx q[3], q[20], q[40];
ccx q[32], q[40], q[38];
ccx q[3], q[20], q[40];
x q[3];
x q[32];
ccx q[38], q[37], q[39];
x q[3];
x q[37];
ccx q[20], q[40], q[37];
ccx q[3], q[4], q[40];
ccx q[20], q[40], q[37];
ccx q[3], q[4], q[40];
x q[3];
x q[25];
x q[32];
x q[40];
ccx q[32], q[39], q[40];
ccx q[24], q[25], q[39];
ccx q[32], q[39], q[40];
ccx q[24], q[25], q[39];
x q[25];
x q[32];
ccx q[39], q[40], q[37];
ccx q[38], q[37], q[39];
x q[3];
x q[32];
x q[38];
ccx q[32], q[40], q[38];
ccx q[3], q[20], q[40];
ccx q[32], q[40], q[38];
ccx q[3], q[20], q[40];
x q[3];
x q[32];
ccx q[38], q[37], q[39];
x q[3];
x q[37];
ccx q[20], q[40], q[37];
ccx q[3], q[4], q[40];
ccx q[20], q[40], q[37];
ccx q[3], q[4], q[40];
x q[3];
ccx q[38], q[37], q[39];
x q[3];
x q[32];
x q[38];
ccx q[32], q[40], q[38];
ccx q[3], q[20], q[40];
ccx q[32], q[40], q[38];
ccx q[3], q[20], q[40];
x q[3];
x q[32];
ccx q[38], q[37], q[39];
x q[3];
x q[37];
ccx q[20], q[40], q[37];
ccx q[3], q[4], q[40];
ccx q[20], q[40], q[37];
ccx q[3], q[4], q[40];
x q[3];
x q[25];
x q[32];
x q[40];
ccx q[32], q[39], q[40];
ccx q[24], q[25], q[39];
ccx q[32], q[39], q[40];
ccx q[24], q[25], q[39];
x q[25];
x q[32];
