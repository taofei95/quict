OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
ccx q[13], q[14], q[11];
ccx q[12], q[11], q[13];
ccx q[13], q[14], q[12];
x q[6];
x q[10];
x q[13];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
x q[6];
x q[10];
ccx q[13], q[14], q[12];
x q[5];
x q[14];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
x q[5];
ccx q[13], q[14], q[12];
x q[6];
x q[10];
x q[13];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
x q[6];
x q[10];
ccx q[13], q[14], q[12];
x q[5];
x q[14];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
x q[5];
ccx q[12], q[11], q[13];
x q[4];
x q[11];
ccx q[4], q[14], q[11];
ccx q[0], q[1], q[14];
ccx q[4], q[14], q[11];
ccx q[0], q[1], q[14];
x q[4];
ccx q[12], q[11], q[13];
ccx q[13], q[14], q[12];
x q[6];
x q[10];
x q[13];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
x q[6];
x q[10];
ccx q[13], q[14], q[12];
x q[5];
x q[14];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
x q[5];
ccx q[13], q[14], q[12];
x q[6];
x q[10];
x q[13];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
x q[6];
x q[10];
ccx q[13], q[14], q[12];
x q[5];
x q[14];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
x q[5];
ccx q[12], q[11], q[13];
x q[4];
x q[11];
ccx q[4], q[14], q[11];
ccx q[0], q[1], q[14];
ccx q[4], q[14], q[11];
ccx q[0], q[1], q[14];
x q[4];
ccx q[13], q[14], q[11];
ccx q[12], q[11], q[14];
x q[6];
x q[12];
ccx q[6], q[14], q[12];
ccx q[4], q[5], q[14];
ccx q[6], q[14], q[12];
ccx q[4], q[5], q[14];
x q[6];
ccx q[12], q[11], q[14];
x q[11];
ccx q[10], q[14], q[11];
ccx q[6], q[9], q[14];
ccx q[10], q[14], q[11];
ccx q[6], q[9], q[14];
ccx q[12], q[11], q[14];
x q[6];
x q[12];
ccx q[6], q[14], q[12];
ccx q[4], q[5], q[14];
ccx q[6], q[14], q[12];
ccx q[4], q[5], q[14];
x q[6];
ccx q[12], q[11], q[14];
x q[11];
ccx q[10], q[14], q[11];
ccx q[6], q[9], q[14];
ccx q[10], q[14], q[11];
ccx q[6], q[9], q[14];
ccx q[13], q[14], q[11];
ccx q[12], q[11], q[13];
ccx q[13], q[14], q[12];
x q[6];
x q[10];
x q[13];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
x q[6];
x q[10];
ccx q[13], q[14], q[12];
x q[5];
x q[14];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
x q[5];
ccx q[13], q[14], q[12];
x q[6];
x q[10];
x q[13];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
x q[6];
x q[10];
ccx q[13], q[14], q[12];
x q[5];
x q[14];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
x q[5];
ccx q[12], q[11], q[13];
x q[4];
x q[11];
ccx q[4], q[14], q[11];
ccx q[0], q[1], q[14];
ccx q[4], q[14], q[11];
ccx q[0], q[1], q[14];
x q[4];
ccx q[12], q[11], q[13];
ccx q[13], q[14], q[12];
x q[6];
x q[10];
x q[13];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
x q[6];
x q[10];
ccx q[13], q[14], q[12];
x q[5];
x q[14];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
x q[5];
ccx q[13], q[14], q[12];
x q[6];
x q[10];
x q[13];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
ccx q[10], q[14], q[13];
ccx q[6], q[8], q[14];
x q[6];
x q[10];
ccx q[13], q[14], q[12];
x q[5];
x q[14];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
ccx q[8], q[13], q[14];
ccx q[5], q[6], q[13];
x q[5];
ccx q[12], q[11], q[13];
x q[4];
x q[11];
ccx q[4], q[14], q[11];
ccx q[0], q[1], q[14];
ccx q[4], q[14], q[11];
ccx q[0], q[1], q[14];
x q[4];
ccx q[13], q[14], q[11];
ccx q[12], q[11], q[14];
x q[6];
x q[12];
ccx q[6], q[14], q[12];
ccx q[4], q[5], q[14];
ccx q[6], q[14], q[12];
ccx q[4], q[5], q[14];
x q[6];
ccx q[12], q[11], q[14];
x q[11];
ccx q[10], q[14], q[11];
ccx q[6], q[9], q[14];
ccx q[10], q[14], q[11];
ccx q[6], q[9], q[14];
ccx q[12], q[11], q[14];
x q[6];
x q[12];
ccx q[6], q[14], q[12];
ccx q[4], q[5], q[14];
ccx q[6], q[14], q[12];
ccx q[4], q[5], q[14];
x q[6];
ccx q[12], q[11], q[14];
x q[11];
ccx q[10], q[14], q[11];
ccx q[6], q[9], q[14];
ccx q[10], q[14], q[11];
ccx q[6], q[9], q[14];