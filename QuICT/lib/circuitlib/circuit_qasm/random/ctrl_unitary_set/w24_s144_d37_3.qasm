OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
creg c[24];
ch q[3], q[10];
cx q[2], q[12];
ch q[13], q[7];
cy q[19], q[14];
cu3(1.5707963267948966, 0, 0) q[14], q[23];
cy q[17], q[22];
cx q[2], q[3];
cy q[8], q[20];
cu3(1.5707963267948966, 0, 0) q[4], q[6];
cx q[14], q[9];
cu3(1.5707963267948966, 0, 0) q[6], q[21];
cy q[14], q[17];
cy q[10], q[21];
ch q[12], q[5];
cu3(1.5707963267948966, 0, 0) q[19], q[16];
cy q[15], q[21];
cx q[8], q[16];
cu3(1.5707963267948966, 0, 0) q[23], q[18];
cy q[3], q[14];
cy q[23], q[11];
ch q[20], q[4];
cy q[17], q[2];
cy q[16], q[11];
cx q[0], q[20];
ch q[2], q[23];
cx q[2], q[20];
cu3(1.5707963267948966, 0, 0) q[8], q[15];
cu3(1.5707963267948966, 0, 0) q[11], q[3];
cy q[6], q[11];
cx q[22], q[19];
cy q[12], q[4];
cy q[6], q[0];
cx q[6], q[4];
cu3(1.5707963267948966, 0, 0) q[12], q[6];
cu3(1.5707963267948966, 0, 0) q[1], q[16];
ch q[19], q[10];
cx q[3], q[12];
cx q[17], q[0];
cx q[7], q[5];
ch q[19], q[17];
cx q[8], q[19];
ch q[5], q[17];
cu3(1.5707963267948966, 0, 0) q[7], q[16];
cy q[22], q[9];
ch q[6], q[11];
ch q[8], q[11];
cu3(1.5707963267948966, 0, 0) q[2], q[7];
cx q[7], q[6];
cy q[4], q[17];
cx q[5], q[21];
cy q[8], q[9];
ch q[10], q[16];
ch q[6], q[2];
cy q[17], q[22];
cx q[2], q[10];
ch q[13], q[5];
ch q[14], q[5];
cy q[1], q[2];
cx q[17], q[11];
cu3(1.5707963267948966, 0, 0) q[15], q[5];
ch q[8], q[10];
cy q[0], q[21];
ch q[3], q[23];
cx q[5], q[19];
cu3(1.5707963267948966, 0, 0) q[15], q[12];
cu3(1.5707963267948966, 0, 0) q[9], q[0];
cu3(1.5707963267948966, 0, 0) q[17], q[4];
cu3(1.5707963267948966, 0, 0) q[23], q[8];
cx q[1], q[19];
cy q[23], q[21];
cy q[7], q[4];
cy q[1], q[14];
cu3(1.5707963267948966, 0, 0) q[6], q[8];
cx q[19], q[13];
cx q[17], q[2];
cx q[16], q[17];
cu3(1.5707963267948966, 0, 0) q[3], q[14];
cy q[21], q[3];
cy q[20], q[15];
ch q[2], q[19];
ch q[5], q[8];
cy q[22], q[17];
cx q[19], q[13];
cx q[13], q[1];
cy q[21], q[13];
ch q[22], q[9];
cx q[18], q[15];
cu3(1.5707963267948966, 0, 0) q[20], q[21];
ch q[15], q[22];
cx q[16], q[20];
cu3(1.5707963267948966, 0, 0) q[0], q[16];
cu3(1.5707963267948966, 0, 0) q[16], q[3];
cx q[0], q[3];
ch q[11], q[21];
cy q[11], q[3];
cy q[17], q[9];
ch q[10], q[6];
cx q[9], q[17];
ch q[10], q[15];
cy q[2], q[20];
cx q[0], q[13];
ch q[15], q[17];
cu3(1.5707963267948966, 0, 0) q[6], q[12];
cx q[5], q[7];
cu3(1.5707963267948966, 0, 0) q[3], q[6];
cy q[16], q[6];
cx q[11], q[4];
ch q[7], q[16];
cy q[7], q[18];
cu3(1.5707963267948966, 0, 0) q[17], q[13];
cx q[10], q[22];
ch q[2], q[23];
cu3(1.5707963267948966, 0, 0) q[13], q[21];
cx q[15], q[8];
cu3(1.5707963267948966, 0, 0) q[10], q[13];
cx q[4], q[19];
ch q[6], q[4];
cy q[22], q[10];
cx q[12], q[14];
ch q[11], q[4];
cy q[4], q[23];
cx q[19], q[9];
cu3(1.5707963267948966, 0, 0) q[9], q[15];
ch q[23], q[22];
cy q[0], q[20];
cx q[17], q[0];
cu3(1.5707963267948966, 0, 0) q[4], q[19];
ch q[19], q[22];
cy q[22], q[21];
cy q[9], q[16];
ch q[4], q[18];
cx q[0], q[10];
cy q[6], q[13];
cy q[14], q[16];
ch q[7], q[3];
ch q[19], q[8];
ch q[8], q[17];
cu3(1.5707963267948966, 0, 0) q[6], q[12];
cu3(1.5707963267948966, 0, 0) q[10], q[2];
cx q[20], q[12];
cy q[7], q[6];
cy q[5], q[16];
cx q[1], q[20];
cx q[16], q[23];
