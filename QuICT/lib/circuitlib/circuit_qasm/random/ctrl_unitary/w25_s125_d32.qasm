OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
cx q[24], q[6];
cx q[3], q[22];
cx q[3], q[5];
cx q[11], q[10];
ch q[14], q[10];
cx q[16], q[4];
cu3(1.5707963267948966, 0, 0) q[23], q[7];
cu3(1.5707963267948966, 0, 0) q[6], q[5];
ch q[9], q[4];
cu3(1.5707963267948966, 0, 0) q[4], q[9];
cu3(1.5707963267948966, 0, 0) q[11], q[20];
cy q[15], q[7];
cy q[4], q[22];
cy q[13], q[10];
cx q[24], q[17];
ch q[3], q[9];
cx q[21], q[14];
cu3(1.5707963267948966, 0, 0) q[23], q[2];
cy q[18], q[14];
ch q[0], q[1];
cy q[19], q[12];
cu3(1.5707963267948966, 0, 0) q[12], q[5];
cu3(1.5707963267948966, 0, 0) q[15], q[23];
cu3(1.5707963267948966, 0, 0) q[1], q[10];
cx q[23], q[5];
cu3(1.5707963267948966, 0, 0) q[9], q[18];
cx q[17], q[10];
cu3(1.5707963267948966, 0, 0) q[2], q[10];
ch q[14], q[23];
ch q[3], q[8];
cu3(1.5707963267948966, 0, 0) q[7], q[5];
cx q[9], q[3];
cy q[5], q[20];
cy q[10], q[5];
cx q[7], q[15];
cx q[10], q[6];
cu3(1.5707963267948966, 0, 0) q[2], q[7];
cu3(1.5707963267948966, 0, 0) q[4], q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[4];
ch q[15], q[17];
ch q[5], q[21];
cx q[17], q[16];
cy q[20], q[13];
ch q[23], q[18];
cy q[4], q[12];
ch q[10], q[2];
ch q[4], q[18];
cu3(1.5707963267948966, 0, 0) q[20], q[3];
ch q[14], q[12];
cu3(1.5707963267948966, 0, 0) q[23], q[24];
cx q[17], q[8];
cy q[5], q[13];
cu3(1.5707963267948966, 0, 0) q[9], q[8];
cu3(1.5707963267948966, 0, 0) q[22], q[24];
cu3(1.5707963267948966, 0, 0) q[9], q[0];
cx q[5], q[3];
ch q[11], q[2];
cx q[17], q[15];
cy q[19], q[22];
cu3(1.5707963267948966, 0, 0) q[17], q[20];
cu3(1.5707963267948966, 0, 0) q[9], q[15];
cu3(1.5707963267948966, 0, 0) q[14], q[3];
cy q[20], q[23];
cy q[11], q[24];
cy q[23], q[18];
cx q[7], q[2];
cy q[2], q[3];
ch q[10], q[18];
cu3(1.5707963267948966, 0, 0) q[12], q[13];
cu3(1.5707963267948966, 0, 0) q[22], q[6];
cu3(1.5707963267948966, 0, 0) q[18], q[24];
ch q[15], q[21];
cx q[15], q[16];
ch q[4], q[13];
cy q[10], q[18];
cu3(1.5707963267948966, 0, 0) q[13], q[14];
cx q[13], q[8];
ch q[10], q[19];
cx q[18], q[4];
cy q[9], q[18];
cx q[7], q[5];
cu3(1.5707963267948966, 0, 0) q[8], q[18];
cx q[15], q[12];
ch q[12], q[21];
cy q[18], q[3];
cu3(1.5707963267948966, 0, 0) q[19], q[5];
cu3(1.5707963267948966, 0, 0) q[9], q[0];
cy q[2], q[21];
cx q[13], q[15];
cu3(1.5707963267948966, 0, 0) q[17], q[16];
cx q[3], q[12];
cy q[24], q[0];
cx q[17], q[6];
ch q[18], q[19];
cy q[7], q[20];
cu3(1.5707963267948966, 0, 0) q[19], q[6];
cy q[5], q[6];
cy q[19], q[2];
cu3(1.5707963267948966, 0, 0) q[3], q[19];
cy q[24], q[19];
cx q[1], q[4];
ch q[5], q[16];
cx q[10], q[2];
cx q[22], q[21];
cy q[20], q[3];
cy q[17], q[12];
ch q[21], q[2];
cx q[18], q[15];
cy q[5], q[23];
cu3(1.5707963267948966, 0, 0) q[23], q[15];
cy q[19], q[22];
cy q[23], q[15];
cx q[23], q[4];
ch q[22], q[9];
cx q[12], q[14];
cx q[13], q[22];
cx q[21], q[5];
cu3(1.5707963267948966, 0, 0) q[13], q[18];
ch q[19], q[10];
cu3(1.5707963267948966, 0, 0) q[7], q[4];
cu3(1.5707963267948966, 0, 0) q[22], q[13];
cy q[6], q[20];
cu3(1.5707963267948966, 0, 0) q[10], q[20];
ch q[13], q[23];
cu3(1.5707963267948966, 0, 0) q[3], q[11];
