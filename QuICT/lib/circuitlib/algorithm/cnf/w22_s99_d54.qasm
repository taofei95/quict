OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
ccx q[19], q[18], q[20];
x q[1];
x q[2];
x q[19];
ccx q[17], q[21], q[19];
ccx q[1], q[2], q[21];
ccx q[17], q[21], q[19];
ccx q[1], q[2], q[21];
x q[1];
x q[2];
ccx q[19], q[18], q[20];
x q[3];
x q[13];
x q[18];
ccx q[13], q[21], q[18];
ccx q[1], q[3], q[21];
ccx q[13], q[21], q[18];
ccx q[1], q[3], q[21];
x q[3];
x q[13];
ccx q[19], q[18], q[20];
x q[1];
x q[2];
x q[19];
ccx q[17], q[21], q[19];
ccx q[1], q[2], q[21];
ccx q[17], q[21], q[19];
ccx q[1], q[2], q[21];
x q[1];
x q[2];
ccx q[19], q[18], q[20];
x q[3];
x q[13];
x q[18];
ccx q[13], q[21], q[18];
ccx q[1], q[3], q[21];
ccx q[13], q[21], q[18];
ccx q[1], q[3], q[21];
x q[3];
x q[13];
x q[8];
x q[15];
x q[21];
ccx q[16], q[20], q[21];
ccx q[8], q[15], q[20];
ccx q[16], q[20], q[21];
ccx q[8], q[15], q[20];
x q[8];
x q[15];
ccx q[20], q[21], q[18];
ccx q[19], q[18], q[20];
x q[1];
x q[2];
x q[19];
ccx q[17], q[21], q[19];
ccx q[1], q[2], q[21];
ccx q[17], q[21], q[19];
ccx q[1], q[2], q[21];
x q[1];
x q[2];
ccx q[19], q[18], q[20];
x q[3];
x q[13];
x q[18];
ccx q[13], q[21], q[18];
ccx q[1], q[3], q[21];
ccx q[13], q[21], q[18];
ccx q[1], q[3], q[21];
x q[3];
x q[13];
ccx q[19], q[18], q[20];
x q[1];
x q[2];
x q[19];
ccx q[17], q[21], q[19];
ccx q[1], q[2], q[21];
ccx q[17], q[21], q[19];
ccx q[1], q[2], q[21];
x q[1];
x q[2];
ccx q[19], q[18], q[20];
x q[3];
x q[13];
x q[18];
ccx q[13], q[21], q[18];
ccx q[1], q[3], q[21];
ccx q[13], q[21], q[18];
ccx q[1], q[3], q[21];
x q[3];
x q[13];
x q[8];
x q[15];
x q[21];
ccx q[16], q[20], q[21];
ccx q[8], q[15], q[20];
ccx q[16], q[20], q[21];
ccx q[8], q[15], q[20];
x q[8];
x q[15];