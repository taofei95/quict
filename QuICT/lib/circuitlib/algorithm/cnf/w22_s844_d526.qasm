OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
ccx q[20], q[21], q[18];
ccx q[19], q[18], q[20];
ccx q[20], q[21], q[19];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[20], q[21], q[19];
x q[5];
x q[8];
x q[21];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
x q[5];
x q[8];
ccx q[20], q[21], q[19];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[20], q[21], q[19];
x q[5];
x q[8];
x q[21];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
x q[5];
x q[8];
ccx q[19], q[18], q[20];
ccx q[20], q[21], q[18];
x q[10];
x q[14];
x q[17];
x q[20];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
x q[10];
x q[14];
x q[17];
ccx q[20], q[21], q[18];
x q[4];
x q[11];
x q[21];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
x q[4];
x q[11];
ccx q[20], q[21], q[18];
x q[10];
x q[14];
x q[17];
x q[20];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
x q[10];
x q[14];
x q[17];
ccx q[20], q[21], q[18];
x q[4];
x q[11];
x q[21];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
x q[4];
x q[11];
ccx q[19], q[18], q[20];
ccx q[20], q[21], q[19];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[20], q[21], q[19];
x q[5];
x q[8];
x q[21];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
x q[5];
x q[8];
ccx q[20], q[21], q[19];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[20], q[21], q[19];
x q[5];
x q[8];
x q[21];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
x q[5];
x q[8];
ccx q[19], q[18], q[20];
ccx q[20], q[21], q[18];
x q[10];
x q[14];
x q[17];
x q[20];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
x q[10];
x q[14];
x q[17];
ccx q[20], q[21], q[18];
x q[4];
x q[11];
x q[21];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
x q[4];
x q[11];
ccx q[20], q[21], q[18];
x q[10];
x q[14];
x q[17];
x q[20];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
x q[10];
x q[14];
x q[17];
ccx q[20], q[21], q[18];
x q[4];
x q[11];
x q[21];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
x q[4];
x q[11];
ccx q[20], q[21], q[18];
ccx q[19], q[18], q[21];
ccx q[20], q[21], q[19];
x q[7];
x q[12];
x q[20];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
x q[7];
x q[12];
ccx q[20], q[21], q[19];
x q[1];
x q[10];
x q[21];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
x q[1];
x q[10];
ccx q[20], q[21], q[19];
x q[7];
x q[12];
x q[20];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
x q[7];
x q[12];
ccx q[20], q[21], q[19];
x q[1];
x q[10];
x q[21];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
x q[1];
x q[10];
ccx q[19], q[18], q[21];
ccx q[20], q[21], q[18];
x q[20];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[20], q[21], q[18];
x q[14];
x q[21];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
x q[14];
ccx q[20], q[21], q[18];
x q[20];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[20], q[21], q[18];
x q[14];
x q[21];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
x q[14];
ccx q[19], q[18], q[21];
ccx q[20], q[21], q[19];
x q[7];
x q[12];
x q[20];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
x q[7];
x q[12];
ccx q[20], q[21], q[19];
x q[1];
x q[10];
x q[21];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
x q[1];
x q[10];
ccx q[20], q[21], q[19];
x q[7];
x q[12];
x q[20];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
x q[7];
x q[12];
ccx q[20], q[21], q[19];
x q[1];
x q[10];
x q[21];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
x q[1];
x q[10];
ccx q[19], q[18], q[21];
ccx q[20], q[21], q[18];
x q[20];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[20], q[21], q[18];
x q[14];
x q[21];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
x q[14];
ccx q[20], q[21], q[18];
x q[20];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[20], q[21], q[18];
x q[14];
x q[21];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
x q[14];
ccx q[20], q[21], q[18];
ccx q[19], q[18], q[20];
ccx q[20], q[21], q[19];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[20], q[21], q[19];
x q[5];
x q[8];
x q[21];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
x q[5];
x q[8];
ccx q[20], q[21], q[19];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[20], q[21], q[19];
x q[5];
x q[8];
x q[21];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
x q[5];
x q[8];
ccx q[19], q[18], q[20];
ccx q[20], q[21], q[18];
x q[10];
x q[14];
x q[17];
x q[20];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
x q[10];
x q[14];
x q[17];
ccx q[20], q[21], q[18];
x q[4];
x q[11];
x q[21];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
x q[4];
x q[11];
ccx q[20], q[21], q[18];
x q[10];
x q[14];
x q[17];
x q[20];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
x q[10];
x q[14];
x q[17];
ccx q[20], q[21], q[18];
x q[4];
x q[11];
x q[21];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
x q[4];
x q[11];
ccx q[19], q[18], q[20];
ccx q[20], q[21], q[19];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[20], q[21], q[19];
x q[5];
x q[8];
x q[21];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
x q[5];
x q[8];
ccx q[20], q[21], q[19];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[19], q[18], q[20];
x q[0];
x q[11];
x q[19];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
ccx q[11], q[21], q[19];
ccx q[0], q[1], q[21];
x q[0];
x q[11];
ccx q[19], q[18], q[20];
x q[17];
x q[18];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
ccx q[17], q[21], q[18];
ccx q[1], q[4], q[21];
x q[17];
ccx q[20], q[21], q[19];
x q[5];
x q[8];
x q[21];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
ccx q[17], q[20], q[21];
ccx q[5], q[8], q[20];
x q[5];
x q[8];
ccx q[19], q[18], q[20];
ccx q[20], q[21], q[18];
x q[10];
x q[14];
x q[17];
x q[20];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
x q[10];
x q[14];
x q[17];
ccx q[20], q[21], q[18];
x q[4];
x q[11];
x q[21];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
x q[4];
x q[11];
ccx q[20], q[21], q[18];
x q[10];
x q[14];
x q[17];
x q[20];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
ccx q[17], q[21], q[20];
ccx q[10], q[14], q[21];
x q[10];
x q[14];
x q[17];
ccx q[20], q[21], q[18];
x q[4];
x q[11];
x q[21];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
ccx q[13], q[20], q[21];
ccx q[4], q[11], q[20];
x q[4];
x q[11];
ccx q[20], q[21], q[18];
ccx q[19], q[18], q[21];
ccx q[20], q[21], q[19];
x q[7];
x q[12];
x q[20];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
x q[7];
x q[12];
ccx q[20], q[21], q[19];
x q[1];
x q[10];
x q[21];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
x q[1];
x q[10];
ccx q[20], q[21], q[19];
x q[7];
x q[12];
x q[20];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
x q[7];
x q[12];
ccx q[20], q[21], q[19];
x q[1];
x q[10];
x q[21];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
x q[1];
x q[10];
ccx q[19], q[18], q[21];
ccx q[20], q[21], q[18];
x q[20];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[20], q[21], q[18];
x q[14];
x q[21];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
x q[14];
ccx q[20], q[21], q[18];
x q[20];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[20], q[21], q[18];
x q[14];
x q[21];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
x q[14];
ccx q[19], q[18], q[21];
ccx q[20], q[21], q[19];
x q[7];
x q[12];
x q[20];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
x q[7];
x q[12];
ccx q[20], q[21], q[19];
x q[1];
x q[10];
x q[21];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
x q[1];
x q[10];
ccx q[20], q[21], q[19];
x q[7];
x q[12];
x q[20];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
ccx q[13], q[21], q[20];
ccx q[7], q[12], q[21];
x q[7];
x q[12];
ccx q[20], q[21], q[19];
x q[1];
x q[10];
x q[21];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
ccx q[10], q[20], q[21];
ccx q[1], q[9], q[20];
x q[1];
x q[10];
ccx q[19], q[18], q[21];
ccx q[20], q[21], q[18];
x q[20];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[20], q[21], q[18];
x q[14];
x q[21];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
x q[14];
ccx q[20], q[21], q[18];
x q[20];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[16], q[21], q[20];
ccx q[3], q[6], q[21];
ccx q[20], q[21], q[18];
x q[14];
x q[21];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
ccx q[14], q[20], q[21];
ccx q[0], q[9], q[20];
x q[14];