OPENQASM 2.0;
include "qelib1.inc";
qreg q[29];
creg c[29];
ch q[19], q[10];
ch q[15], q[3];
cx q[22], q[3];
cx q[28], q[10];
ch q[22], q[5];
cu3(1.5707963267948966, 0, 0) q[9], q[19];
cx q[28], q[6];
ch q[24], q[21];
ch q[9], q[13];
cy q[23], q[26];
cy q[16], q[28];
cx q[10], q[17];
cx q[28], q[11];
cx q[17], q[9];
ch q[1], q[28];
cu3(1.5707963267948966, 0, 0) q[0], q[14];
cx q[14], q[0];
cu3(1.5707963267948966, 0, 0) q[19], q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[5];
cu3(1.5707963267948966, 0, 0) q[27], q[13];
cx q[28], q[9];
cy q[28], q[25];
cx q[5], q[10];
cy q[25], q[28];
cy q[0], q[25];
ch q[28], q[27];
cx q[13], q[27];
cx q[15], q[23];
cy q[15], q[1];
cu3(1.5707963267948966, 0, 0) q[7], q[12];
cx q[5], q[9];
cx q[15], q[8];
cy q[13], q[11];
cu3(1.5707963267948966, 0, 0) q[15], q[14];
cx q[17], q[6];
cx q[22], q[0];
cx q[23], q[15];
cy q[16], q[14];
cy q[18], q[12];
ch q[12], q[25];
cx q[0], q[6];
cx q[24], q[22];
cy q[3], q[1];
cu3(1.5707963267948966, 0, 0) q[17], q[5];
cy q[15], q[3];
cx q[14], q[20];
ch q[21], q[27];
ch q[14], q[23];
ch q[23], q[5];
cy q[22], q[4];
cx q[0], q[6];
ch q[25], q[22];
cu3(1.5707963267948966, 0, 0) q[12], q[28];
cx q[23], q[21];
cx q[26], q[10];
cx q[15], q[12];
cx q[10], q[3];
cu3(1.5707963267948966, 0, 0) q[16], q[3];
ch q[6], q[25];
cy q[8], q[4];
cx q[23], q[5];
cy q[8], q[14];
cx q[26], q[17];
cy q[1], q[21];
ch q[19], q[1];
ch q[14], q[13];
cx q[22], q[6];
cu3(1.5707963267948966, 0, 0) q[28], q[22];
cx q[0], q[16];
cx q[10], q[14];
ch q[1], q[2];
ch q[25], q[10];
cy q[9], q[7];
ch q[6], q[1];
ch q[26], q[16];
cx q[11], q[13];
cx q[3], q[0];
cx q[1], q[0];
cx q[26], q[8];
cu3(1.5707963267948966, 0, 0) q[27], q[3];
cx q[28], q[14];
cu3(1.5707963267948966, 0, 0) q[23], q[16];
ch q[20], q[23];
ch q[4], q[17];
cy q[21], q[0];
cx q[2], q[18];
ch q[22], q[9];
cx q[0], q[23];
ch q[11], q[1];
cu3(1.5707963267948966, 0, 0) q[26], q[25];
ch q[14], q[28];
cx q[26], q[12];
cu3(1.5707963267948966, 0, 0) q[1], q[6];
cy q[9], q[7];
cy q[21], q[5];
cx q[0], q[24];
cu3(1.5707963267948966, 0, 0) q[26], q[16];
ch q[10], q[18];
cy q[26], q[27];
ch q[0], q[23];
cx q[4], q[9];
cx q[23], q[0];
ch q[25], q[14];
cu3(1.5707963267948966, 0, 0) q[1], q[8];
cy q[5], q[8];
cu3(1.5707963267948966, 0, 0) q[20], q[9];
ch q[26], q[11];
cx q[11], q[1];
cy q[21], q[15];
cu3(1.5707963267948966, 0, 0) q[12], q[24];
cu3(1.5707963267948966, 0, 0) q[5], q[7];
cu3(1.5707963267948966, 0, 0) q[1], q[25];
ch q[16], q[7];
cx q[6], q[4];
cx q[26], q[9];
cx q[0], q[14];
ch q[22], q[12];
cx q[2], q[10];
ch q[28], q[0];
ch q[27], q[14];
cy q[13], q[20];
cy q[28], q[24];
cy q[19], q[22];
cu3(1.5707963267948966, 0, 0) q[23], q[9];
ch q[7], q[19];
cu3(1.5707963267948966, 0, 0) q[21], q[20];
cy q[27], q[15];
ch q[12], q[0];
cu3(1.5707963267948966, 0, 0) q[25], q[23];
ch q[15], q[0];
cu3(1.5707963267948966, 0, 0) q[12], q[15];
cy q[16], q[6];
cu3(1.5707963267948966, 0, 0) q[22], q[5];
cu3(1.5707963267948966, 0, 0) q[22], q[10];
cy q[22], q[18];
ch q[4], q[22];
ch q[1], q[23];
ch q[12], q[6];
cx q[3], q[22];
cx q[8], q[23];
cy q[13], q[23];
cx q[27], q[1];
cu3(1.5707963267948966, 0, 0) q[9], q[7];
cx q[12], q[17];
cx q[21], q[8];
cu3(1.5707963267948966, 0, 0) q[1], q[15];
ch q[15], q[27];
cu3(1.5707963267948966, 0, 0) q[4], q[14];
cu3(1.5707963267948966, 0, 0) q[8], q[17];
cy q[18], q[15];
cu3(1.5707963267948966, 0, 0) q[8], q[3];
cu3(1.5707963267948966, 0, 0) q[25], q[1];
cx q[15], q[4];
cy q[16], q[24];
cx q[10], q[19];
cx q[22], q[6];
ch q[19], q[27];
cx q[11], q[26];
ch q[21], q[15];
cx q[5], q[27];
cy q[23], q[0];
cx q[0], q[20];
cx q[2], q[22];
ch q[12], q[22];
cu3(1.5707963267948966, 0, 0) q[9], q[22];
cx q[1], q[19];
ch q[3], q[27];
cy q[26], q[16];
cu3(1.5707963267948966, 0, 0) q[14], q[23];
ch q[23], q[26];
cx q[8], q[21];
cu3(1.5707963267948966, 0, 0) q[17], q[13];
cu3(1.5707963267948966, 0, 0) q[5], q[11];
cu3(1.5707963267948966, 0, 0) q[9], q[17];
