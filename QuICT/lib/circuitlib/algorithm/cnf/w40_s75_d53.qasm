OPENQASM 2.0;
include "qelib1.inc";
qreg q[40];
creg c[40];
ccx q[37], q[36], q[38];
x q[37];
ccx q[33], q[39], q[37];
ccx q[13], q[30], q[39];
ccx q[33], q[39], q[37];
ccx q[13], q[30], q[39];
ccx q[37], q[36], q[38];
x q[12];
x q[13];
x q[36];
ccx q[13], q[39], q[36];
ccx q[7], q[12], q[39];
ccx q[13], q[39], q[36];
ccx q[7], q[12], q[39];
x q[12];
x q[13];
ccx q[37], q[36], q[38];
x q[37];
ccx q[33], q[39], q[37];
ccx q[13], q[30], q[39];
ccx q[33], q[39], q[37];
ccx q[13], q[30], q[39];
ccx q[37], q[36], q[38];
x q[12];
x q[13];
x q[36];
ccx q[13], q[39], q[36];
ccx q[7], q[12], q[39];
ccx q[13], q[39], q[36];
ccx q[7], q[12], q[39];
x q[12];
x q[13];
x q[39];
ccx q[18], q[38], q[39];
ccx q[2], q[7], q[38];
ccx q[18], q[38], q[39];
ccx q[2], q[7], q[38];
ccx q[38], q[39], q[36];
ccx q[37], q[36], q[38];
x q[37];
ccx q[33], q[39], q[37];
ccx q[13], q[30], q[39];
ccx q[33], q[39], q[37];
ccx q[13], q[30], q[39];
ccx q[37], q[36], q[38];
x q[12];
x q[13];
x q[36];
ccx q[13], q[39], q[36];
ccx q[7], q[12], q[39];
ccx q[13], q[39], q[36];
ccx q[7], q[12], q[39];
x q[12];
x q[13];
ccx q[37], q[36], q[38];
x q[37];
ccx q[33], q[39], q[37];
ccx q[13], q[30], q[39];
ccx q[33], q[39], q[37];
ccx q[13], q[30], q[39];
ccx q[37], q[36], q[38];
x q[12];
x q[13];
x q[36];
ccx q[13], q[39], q[36];
ccx q[7], q[12], q[39];
ccx q[13], q[39], q[36];
ccx q[7], q[12], q[39];
x q[12];
x q[13];
x q[39];
ccx q[18], q[38], q[39];
ccx q[2], q[7], q[38];
ccx q[18], q[38], q[39];
ccx q[2], q[7], q[38];
