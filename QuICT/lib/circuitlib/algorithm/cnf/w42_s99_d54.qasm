OPENQASM 2.0;
include "qelib1.inc";
qreg q[42];
creg c[42];
ccx q[39], q[38], q[40];
x q[0];
x q[27];
x q[33];
x q[39];
ccx q[33], q[41], q[39];
ccx q[0], q[27], q[41];
ccx q[33], q[41], q[39];
ccx q[0], q[27], q[41];
x q[0];
x q[27];
x q[33];
ccx q[39], q[38], q[40];
x q[1];
x q[38];
ccx q[27], q[41], q[38];
ccx q[1], q[6], q[41];
ccx q[27], q[41], q[38];
ccx q[1], q[6], q[41];
x q[1];
ccx q[39], q[38], q[40];
x q[0];
x q[27];
x q[33];
x q[39];
ccx q[33], q[41], q[39];
ccx q[0], q[27], q[41];
ccx q[33], q[41], q[39];
ccx q[0], q[27], q[41];
x q[0];
x q[27];
x q[33];
ccx q[39], q[38], q[40];
x q[1];
x q[38];
ccx q[27], q[41], q[38];
ccx q[1], q[6], q[41];
ccx q[27], q[41], q[38];
ccx q[1], q[6], q[41];
x q[1];
x q[19];
x q[33];
x q[41];
ccx q[33], q[40], q[41];
ccx q[1], q[19], q[40];
ccx q[33], q[40], q[41];
ccx q[1], q[19], q[40];
x q[19];
x q[33];
ccx q[40], q[41], q[38];
ccx q[39], q[38], q[40];
x q[0];
x q[27];
x q[33];
x q[39];
ccx q[33], q[41], q[39];
ccx q[0], q[27], q[41];
ccx q[33], q[41], q[39];
ccx q[0], q[27], q[41];
x q[0];
x q[27];
x q[33];
ccx q[39], q[38], q[40];
x q[1];
x q[38];
ccx q[27], q[41], q[38];
ccx q[1], q[6], q[41];
ccx q[27], q[41], q[38];
ccx q[1], q[6], q[41];
x q[1];
ccx q[39], q[38], q[40];
x q[0];
x q[27];
x q[33];
x q[39];
ccx q[33], q[41], q[39];
ccx q[0], q[27], q[41];
ccx q[33], q[41], q[39];
ccx q[0], q[27], q[41];
x q[0];
x q[27];
x q[33];
ccx q[39], q[38], q[40];
x q[1];
x q[38];
ccx q[27], q[41], q[38];
ccx q[1], q[6], q[41];
ccx q[27], q[41], q[38];
ccx q[1], q[6], q[41];
x q[1];
x q[19];
x q[33];
x q[41];
ccx q[33], q[40], q[41];
ccx q[1], q[19], q[40];
ccx q[33], q[40], q[41];
ccx q[1], q[19], q[40];
x q[19];
x q[33];
