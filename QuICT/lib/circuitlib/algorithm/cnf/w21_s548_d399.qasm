OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
ccx q[19], q[20], q[17];
ccx q[18], q[17], q[19];
ccx q[19], q[20], q[18];
x q[8];
x q[19];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
x q[8];
ccx q[19], q[20], q[18];
x q[7];
x q[14];
x q[20];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
x q[7];
x q[14];
ccx q[19], q[20], q[18];
x q[8];
x q[19];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
x q[8];
ccx q[19], q[20], q[18];
x q[7];
x q[14];
x q[20];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
x q[7];
x q[14];
ccx q[18], q[17], q[19];
ccx q[19], q[20], q[17];
x q[14];
x q[19];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
x q[14];
ccx q[19], q[20], q[17];
x q[20];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[19], q[20], q[17];
x q[14];
x q[19];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
x q[14];
ccx q[19], q[20], q[17];
x q[20];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[18], q[17], q[19];
ccx q[19], q[20], q[18];
x q[8];
x q[19];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
x q[8];
ccx q[19], q[20], q[18];
x q[7];
x q[14];
x q[20];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
x q[7];
x q[14];
ccx q[19], q[20], q[18];
x q[8];
x q[19];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
x q[8];
ccx q[19], q[20], q[18];
x q[7];
x q[14];
x q[20];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
x q[7];
x q[14];
ccx q[18], q[17], q[19];
ccx q[19], q[20], q[17];
x q[14];
x q[19];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
x q[14];
ccx q[19], q[20], q[17];
x q[20];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[19], q[20], q[17];
x q[14];
x q[19];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
x q[14];
ccx q[19], q[20], q[17];
x q[20];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[19], q[20], q[17];
ccx q[18], q[17], q[20];
ccx q[19], q[20], q[18];
x q[12];
x q[19];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[20];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[19];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[20];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
x q[12];
ccx q[18], q[17], q[20];
ccx q[19], q[20], q[17];
x q[13];
x q[19];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
x q[13];
ccx q[19], q[20], q[17];
x q[10];
x q[12];
x q[20];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
x q[10];
x q[12];
ccx q[19], q[20], q[17];
x q[13];
x q[19];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
x q[13];
ccx q[19], q[20], q[17];
x q[10];
x q[12];
x q[20];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
x q[10];
x q[12];
ccx q[18], q[17], q[20];
ccx q[19], q[20], q[18];
x q[12];
x q[19];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[20];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[19];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[20];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
x q[12];
ccx q[18], q[17], q[20];
ccx q[19], q[20], q[17];
x q[13];
x q[19];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
x q[13];
ccx q[19], q[20], q[17];
x q[10];
x q[12];
x q[20];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
x q[10];
x q[12];
ccx q[19], q[20], q[17];
x q[13];
x q[19];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
x q[13];
ccx q[19], q[20], q[17];
x q[10];
x q[12];
x q[20];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
x q[10];
x q[12];
ccx q[19], q[20], q[17];
ccx q[18], q[17], q[19];
ccx q[19], q[20], q[18];
x q[8];
x q[19];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
x q[8];
ccx q[19], q[20], q[18];
x q[7];
x q[14];
x q[20];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
x q[7];
x q[14];
ccx q[19], q[20], q[18];
x q[8];
x q[19];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
x q[8];
ccx q[19], q[20], q[18];
x q[7];
x q[14];
x q[20];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
x q[7];
x q[14];
ccx q[18], q[17], q[19];
ccx q[19], q[20], q[17];
x q[14];
x q[19];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
x q[14];
ccx q[19], q[20], q[17];
x q[20];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[19], q[20], q[17];
x q[14];
x q[19];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
x q[14];
ccx q[19], q[20], q[17];
x q[20];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[18], q[17], q[19];
ccx q[19], q[20], q[18];
x q[8];
x q[19];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
x q[8];
ccx q[19], q[20], q[18];
x q[7];
x q[14];
x q[20];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
x q[7];
x q[14];
ccx q[19], q[20], q[18];
x q[8];
x q[19];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
ccx q[8], q[20], q[19];
ccx q[6], q[7], q[20];
x q[8];
ccx q[19], q[20], q[18];
x q[7];
x q[14];
x q[20];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
ccx q[14], q[19], q[20];
ccx q[7], q[8], q[19];
x q[7];
x q[14];
ccx q[18], q[17], q[19];
ccx q[19], q[20], q[17];
x q[14];
x q[19];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
x q[14];
ccx q[19], q[20], q[17];
x q[20];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[19], q[20], q[17];
x q[14];
x q[19];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
ccx q[14], q[20], q[19];
ccx q[2], q[5], q[20];
x q[14];
ccx q[19], q[20], q[17];
x q[20];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[16], q[19], q[20];
ccx q[3], q[5], q[19];
ccx q[19], q[20], q[17];
ccx q[18], q[17], q[20];
ccx q[19], q[20], q[18];
x q[12];
x q[19];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[20];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[19];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[20];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
x q[12];
ccx q[18], q[17], q[20];
ccx q[19], q[20], q[17];
x q[13];
x q[19];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
x q[13];
ccx q[19], q[20], q[17];
x q[10];
x q[12];
x q[20];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
x q[10];
x q[12];
ccx q[19], q[20], q[17];
x q[13];
x q[19];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
x q[13];
ccx q[19], q[20], q[17];
x q[10];
x q[12];
x q[20];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
x q[10];
x q[12];
ccx q[18], q[17], q[20];
ccx q[19], q[20], q[18];
x q[12];
x q[19];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[20];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[19];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
ccx q[16], q[20], q[19];
ccx q[0], q[12], q[20];
x q[12];
ccx q[19], q[20], q[18];
x q[12];
x q[20];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
ccx q[12], q[19], q[20];
ccx q[0], q[6], q[19];
x q[12];
ccx q[18], q[17], q[20];
ccx q[19], q[20], q[17];
x q[13];
x q[19];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
x q[13];
ccx q[19], q[20], q[17];
x q[10];
x q[12];
x q[20];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
x q[10];
x q[12];
ccx q[19], q[20], q[17];
x q[13];
x q[19];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
ccx q[13], q[20], q[19];
ccx q[10], q[11], q[20];
x q[13];
ccx q[19], q[20], q[17];
x q[10];
x q[12];
x q[20];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
ccx q[15], q[19], q[20];
ccx q[10], q[12], q[19];
x q[10];
x q[12];
