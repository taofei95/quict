OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c[28];
cy q[9], q[17];
ch q[24], q[12];
cu3(1.5707963267948966, 0, 0) q[24], q[21];
cu3(1.5707963267948966, 0, 0) q[9], q[19];
cx q[18], q[5];
cy q[21], q[13];
ch q[13], q[19];
cu3(1.5707963267948966, 0, 0) q[20], q[22];
cu3(1.5707963267948966, 0, 0) q[27], q[9];
ch q[4], q[9];
cu3(1.5707963267948966, 0, 0) q[27], q[6];
cu3(1.5707963267948966, 0, 0) q[20], q[6];
cu3(1.5707963267948966, 0, 0) q[13], q[12];
cx q[12], q[6];
cx q[19], q[16];
cu3(1.5707963267948966, 0, 0) q[17], q[5];
ch q[1], q[26];
cx q[13], q[1];
cu3(1.5707963267948966, 0, 0) q[2], q[18];
ch q[18], q[20];
ch q[15], q[18];
cu3(1.5707963267948966, 0, 0) q[7], q[15];
cu3(1.5707963267948966, 0, 0) q[5], q[12];
cy q[26], q[1];
ch q[1], q[7];
cy q[14], q[1];
cx q[22], q[27];
cu3(1.5707963267948966, 0, 0) q[6], q[3];
cu3(1.5707963267948966, 0, 0) q[24], q[25];
cy q[2], q[24];
cx q[2], q[24];
cy q[19], q[26];
cu3(1.5707963267948966, 0, 0) q[18], q[16];
cu3(1.5707963267948966, 0, 0) q[25], q[13];
cu3(1.5707963267948966, 0, 0) q[24], q[18];
ch q[13], q[3];
cu3(1.5707963267948966, 0, 0) q[27], q[12];
cy q[5], q[2];
cu3(1.5707963267948966, 0, 0) q[26], q[22];
cy q[13], q[4];
cx q[18], q[11];
cy q[18], q[8];
cu3(1.5707963267948966, 0, 0) q[6], q[11];
cy q[17], q[2];
cu3(1.5707963267948966, 0, 0) q[17], q[18];
ch q[27], q[17];
ch q[8], q[19];
cu3(1.5707963267948966, 0, 0) q[18], q[10];
cy q[1], q[4];
cx q[12], q[13];
cx q[2], q[7];
cy q[27], q[26];
cy q[4], q[23];
cx q[16], q[1];
ch q[22], q[5];
cu3(1.5707963267948966, 0, 0) q[14], q[12];
cx q[13], q[3];
cx q[4], q[8];
cx q[22], q[6];
cy q[1], q[4];
cu3(1.5707963267948966, 0, 0) q[5], q[25];
ch q[9], q[2];
cu3(1.5707963267948966, 0, 0) q[13], q[19];
cx q[27], q[21];
cu3(1.5707963267948966, 0, 0) q[12], q[13];
ch q[23], q[12];
cy q[21], q[1];
cu3(1.5707963267948966, 0, 0) q[16], q[12];
cy q[4], q[22];
cu3(1.5707963267948966, 0, 0) q[27], q[2];
cx q[3], q[21];
ch q[9], q[20];
cu3(1.5707963267948966, 0, 0) q[5], q[18];
ch q[11], q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[11];
cx q[21], q[9];
ch q[22], q[27];
cx q[14], q[5];
cy q[1], q[10];
cu3(1.5707963267948966, 0, 0) q[21], q[19];
ch q[11], q[20];
cu3(1.5707963267948966, 0, 0) q[22], q[4];
cu3(1.5707963267948966, 0, 0) q[27], q[0];
cx q[3], q[1];
cu3(1.5707963267948966, 0, 0) q[8], q[27];
cy q[15], q[20];
ch q[10], q[12];
cx q[2], q[3];
cy q[12], q[27];
cy q[24], q[26];
cx q[16], q[5];
ch q[9], q[6];
cy q[14], q[2];
cy q[1], q[15];
ch q[19], q[5];
cy q[18], q[1];
ch q[25], q[26];
cx q[3], q[1];
cx q[17], q[1];
ch q[9], q[11];
cx q[1], q[10];
cy q[17], q[3];
cx q[27], q[14];
cu3(1.5707963267948966, 0, 0) q[8], q[2];
cu3(1.5707963267948966, 0, 0) q[13], q[20];
ch q[12], q[22];
cx q[24], q[8];
cy q[3], q[6];
ch q[9], q[14];
cu3(1.5707963267948966, 0, 0) q[22], q[26];
cx q[2], q[22];
ch q[23], q[5];
cx q[4], q[2];
cu3(1.5707963267948966, 0, 0) q[21], q[4];
cy q[27], q[14];
cy q[16], q[3];
cy q[17], q[23];
ch q[27], q[11];
cy q[3], q[13];
cx q[18], q[7];
cu3(1.5707963267948966, 0, 0) q[19], q[16];
cy q[8], q[15];
cu3(1.5707963267948966, 0, 0) q[0], q[3];
cu3(1.5707963267948966, 0, 0) q[23], q[18];
cx q[4], q[24];
cy q[27], q[13];
cy q[16], q[22];
cx q[4], q[26];
cu3(1.5707963267948966, 0, 0) q[18], q[16];
cy q[25], q[17];
cy q[17], q[21];
cy q[23], q[10];
ch q[21], q[25];
cx q[1], q[12];
cy q[15], q[19];
cy q[6], q[23];
cu3(1.5707963267948966, 0, 0) q[26], q[25];
cx q[1], q[9];
ch q[17], q[4];
cy q[11], q[14];
