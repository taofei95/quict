OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
ccx q[17], q[16], q[18];
x q[17];
ccx q[11], q[19], q[17];
ccx q[4], q[6], q[19];
ccx q[11], q[19], q[17];
ccx q[4], q[6], q[19];
ccx q[17], q[16], q[18];
x q[2];
x q[11];
x q[16];
ccx q[11], q[19], q[16];
ccx q[2], q[5], q[19];
ccx q[11], q[19], q[16];
ccx q[2], q[5], q[19];
x q[2];
x q[11];
ccx q[17], q[16], q[18];
x q[17];
ccx q[11], q[19], q[17];
ccx q[4], q[6], q[19];
ccx q[11], q[19], q[17];
ccx q[4], q[6], q[19];
ccx q[17], q[16], q[18];
x q[2];
x q[11];
x q[16];
ccx q[11], q[19], q[16];
ccx q[2], q[5], q[19];
ccx q[11], q[19], q[16];
ccx q[2], q[5], q[19];
x q[2];
x q[11];
x q[0];
x q[12];
x q[19];
ccx q[12], q[18], q[19];
ccx q[0], q[9], q[18];
ccx q[12], q[18], q[19];
ccx q[0], q[9], q[18];
x q[0];
x q[12];
ccx q[18], q[19], q[16];
ccx q[17], q[16], q[18];
x q[17];
ccx q[11], q[19], q[17];
ccx q[4], q[6], q[19];
ccx q[11], q[19], q[17];
ccx q[4], q[6], q[19];
ccx q[17], q[16], q[18];
x q[2];
x q[11];
x q[16];
ccx q[11], q[19], q[16];
ccx q[2], q[5], q[19];
ccx q[11], q[19], q[16];
ccx q[2], q[5], q[19];
x q[2];
x q[11];
ccx q[17], q[16], q[18];
x q[17];
ccx q[11], q[19], q[17];
ccx q[4], q[6], q[19];
ccx q[11], q[19], q[17];
ccx q[4], q[6], q[19];
ccx q[17], q[16], q[18];
x q[2];
x q[11];
x q[16];
ccx q[11], q[19], q[16];
ccx q[2], q[5], q[19];
ccx q[11], q[19], q[16];
ccx q[2], q[5], q[19];
x q[2];
x q[11];
x q[0];
x q[12];
x q[19];
ccx q[12], q[18], q[19];
ccx q[0], q[9], q[18];
ccx q[12], q[18], q[19];
ccx q[0], q[9], q[18];
x q[0];
x q[12];