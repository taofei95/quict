OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
ccx q[23], q[24], q[21];
ccx q[22], q[21], q[23];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[23], q[24], q[22];
x q[2];
x q[9];
x q[24];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
x q[2];
x q[9];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[23], q[24], q[22];
x q[2];
x q[9];
x q[24];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
x q[2];
x q[9];
ccx q[22], q[21], q[23];
ccx q[23], q[24], q[21];
x q[0];
x q[16];
x q[23];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
x q[0];
x q[16];
ccx q[23], q[24], q[21];
x q[3];
x q[20];
x q[24];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
x q[3];
x q[20];
ccx q[23], q[24], q[21];
x q[0];
x q[16];
x q[23];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
x q[0];
x q[16];
ccx q[23], q[24], q[21];
x q[3];
x q[20];
x q[24];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
x q[3];
x q[20];
ccx q[22], q[21], q[23];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[23], q[24], q[22];
x q[2];
x q[9];
x q[24];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
x q[2];
x q[9];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[23], q[24], q[22];
x q[2];
x q[9];
x q[24];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
x q[2];
x q[9];
ccx q[22], q[21], q[23];
ccx q[23], q[24], q[21];
x q[0];
x q[16];
x q[23];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
x q[0];
x q[16];
ccx q[23], q[24], q[21];
x q[3];
x q[20];
x q[24];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
x q[3];
x q[20];
ccx q[23], q[24], q[21];
x q[0];
x q[16];
x q[23];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
x q[0];
x q[16];
ccx q[23], q[24], q[21];
x q[3];
x q[20];
x q[24];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
x q[3];
x q[20];
ccx q[23], q[24], q[21];
ccx q[22], q[21], q[24];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[23], q[24], q[22];
x q[15];
x q[24];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
x q[15];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[23], q[24], q[22];
x q[15];
x q[24];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
x q[15];
ccx q[22], q[21], q[24];
ccx q[23], q[24], q[21];
x q[10];
x q[23];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
x q[10];
ccx q[23], q[24], q[21];
x q[9];
x q[19];
x q[24];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
x q[9];
x q[19];
ccx q[23], q[24], q[21];
x q[10];
x q[23];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
x q[10];
ccx q[23], q[24], q[21];
x q[9];
x q[19];
x q[24];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
x q[9];
x q[19];
ccx q[22], q[21], q[24];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[23], q[24], q[22];
x q[15];
x q[24];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
x q[15];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[23], q[24], q[22];
x q[15];
x q[24];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
x q[15];
ccx q[22], q[21], q[24];
ccx q[23], q[24], q[21];
x q[10];
x q[23];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
x q[10];
ccx q[23], q[24], q[21];
x q[9];
x q[19];
x q[24];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
x q[9];
x q[19];
ccx q[23], q[24], q[21];
x q[10];
x q[23];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
x q[10];
ccx q[23], q[24], q[21];
x q[9];
x q[19];
x q[24];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
x q[9];
x q[19];
ccx q[23], q[24], q[21];
ccx q[22], q[21], q[23];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[23], q[24], q[22];
x q[2];
x q[9];
x q[24];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
x q[2];
x q[9];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[23], q[24], q[22];
x q[2];
x q[9];
x q[24];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
x q[2];
x q[9];
ccx q[22], q[21], q[23];
ccx q[23], q[24], q[21];
x q[0];
x q[16];
x q[23];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
x q[0];
x q[16];
ccx q[23], q[24], q[21];
x q[3];
x q[20];
x q[24];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
x q[3];
x q[20];
ccx q[23], q[24], q[21];
x q[0];
x q[16];
x q[23];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
x q[0];
x q[16];
ccx q[23], q[24], q[21];
x q[3];
x q[20];
x q[24];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
x q[3];
x q[20];
ccx q[22], q[21], q[23];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[23], q[24], q[22];
x q[2];
x q[9];
x q[24];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
x q[2];
x q[9];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[22], q[21], q[23];
x q[0];
x q[5];
x q[7];
x q[22];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
ccx q[7], q[24], q[22];
ccx q[0], q[5], q[24];
x q[0];
x q[5];
x q[7];
ccx q[22], q[21], q[23];
x q[1];
x q[11];
x q[21];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
ccx q[15], q[24], q[21];
ccx q[1], q[11], q[24];
x q[1];
x q[11];
ccx q[23], q[24], q[22];
x q[2];
x q[9];
x q[24];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
ccx q[9], q[23], q[24];
ccx q[2], q[7], q[23];
x q[2];
x q[9];
ccx q[22], q[21], q[23];
ccx q[23], q[24], q[21];
x q[0];
x q[16];
x q[23];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
x q[0];
x q[16];
ccx q[23], q[24], q[21];
x q[3];
x q[20];
x q[24];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
x q[3];
x q[20];
ccx q[23], q[24], q[21];
x q[0];
x q[16];
x q[23];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
ccx q[16], q[24], q[23];
ccx q[0], q[8], q[24];
x q[0];
x q[16];
ccx q[23], q[24], q[21];
x q[3];
x q[20];
x q[24];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
ccx q[20], q[23], q[24];
ccx q[3], q[16], q[23];
x q[3];
x q[20];
ccx q[23], q[24], q[21];
ccx q[22], q[21], q[24];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[23], q[24], q[22];
x q[15];
x q[24];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
x q[15];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[23], q[24], q[22];
x q[15];
x q[24];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
x q[15];
ccx q[22], q[21], q[24];
ccx q[23], q[24], q[21];
x q[10];
x q[23];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
x q[10];
ccx q[23], q[24], q[21];
x q[9];
x q[19];
x q[24];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
x q[9];
x q[19];
ccx q[23], q[24], q[21];
x q[10];
x q[23];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
x q[10];
ccx q[23], q[24], q[21];
x q[9];
x q[19];
x q[24];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
x q[9];
x q[19];
ccx q[22], q[21], q[24];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[23], q[24], q[22];
x q[15];
x q[24];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
x q[15];
ccx q[23], q[24], q[22];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[22];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
ccx q[8], q[24], q[22];
ccx q[3], q[4], q[24];
x q[8];
ccx q[22], q[21], q[23];
x q[8];
x q[21];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
ccx q[13], q[24], q[21];
ccx q[8], q[11], q[24];
x q[8];
ccx q[23], q[24], q[22];
x q[15];
x q[24];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
ccx q[15], q[23], q[24];
ccx q[3], q[9], q[23];
x q[15];
ccx q[22], q[21], q[24];
ccx q[23], q[24], q[21];
x q[10];
x q[23];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
x q[10];
ccx q[23], q[24], q[21];
x q[9];
x q[19];
x q[24];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
x q[9];
x q[19];
ccx q[23], q[24], q[21];
x q[10];
x q[23];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
ccx q[11], q[24], q[23];
ccx q[2], q[10], q[24];
x q[10];
ccx q[23], q[24], q[21];
x q[9];
x q[19];
x q[24];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
ccx q[19], q[23], q[24];
ccx q[5], q[9], q[23];
x q[9];
x q[19];
