OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
cy q[14], q[1];
cu3(1.5707963267948966, 0, 0) q[2], q[6];
ch q[9], q[8];
cu3(1.5707963267948966, 0, 0) q[18], q[8];
cy q[21], q[1];
ch q[3], q[6];
ch q[3], q[13];
cx q[13], q[9];
cu3(1.5707963267948966, 0, 0) q[18], q[22];
cu3(1.5707963267948966, 0, 0) q[4], q[20];
ch q[7], q[3];
cx q[17], q[10];
cy q[11], q[14];
ch q[7], q[21];
cy q[22], q[19];
ch q[22], q[7];
cy q[9], q[16];
cu3(1.5707963267948966, 0, 0) q[0], q[3];
ch q[9], q[20];
ch q[4], q[3];
cy q[18], q[11];
cy q[18], q[19];
cx q[9], q[10];
cu3(1.5707963267948966, 0, 0) q[21], q[14];
cx q[9], q[19];
ch q[5], q[16];
cy q[13], q[20];
ch q[10], q[7];
cx q[2], q[19];
cy q[15], q[0];
cu3(1.5707963267948966, 0, 0) q[4], q[12];
cu3(1.5707963267948966, 0, 0) q[16], q[2];
cy q[17], q[10];
ch q[5], q[22];
cy q[22], q[18];
cx q[17], q[19];
cy q[13], q[10];
cu3(1.5707963267948966, 0, 0) q[3], q[12];
cx q[3], q[16];
cu3(1.5707963267948966, 0, 0) q[9], q[5];
ch q[22], q[1];
cu3(1.5707963267948966, 0, 0) q[2], q[9];
cu3(1.5707963267948966, 0, 0) q[7], q[19];
ch q[5], q[22];
ch q[10], q[5];
ch q[16], q[12];
ch q[9], q[11];
cu3(1.5707963267948966, 0, 0) q[14], q[2];
cx q[10], q[12];
cy q[20], q[21];
cy q[17], q[15];
ch q[7], q[6];
ch q[10], q[22];
ch q[0], q[7];
ch q[11], q[9];
ch q[13], q[16];
cu3(1.5707963267948966, 0, 0) q[14], q[16];
cx q[15], q[16];
cx q[10], q[4];
cu3(1.5707963267948966, 0, 0) q[21], q[11];
cu3(1.5707963267948966, 0, 0) q[18], q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[17];
ch q[21], q[22];
cy q[17], q[7];
cu3(1.5707963267948966, 0, 0) q[20], q[3];
cy q[14], q[2];
cx q[6], q[18];
ch q[14], q[2];
cx q[8], q[7];
cx q[21], q[15];
ch q[6], q[0];
ch q[7], q[18];
cu3(1.5707963267948966, 0, 0) q[15], q[20];
cx q[16], q[19];
cu3(1.5707963267948966, 0, 0) q[7], q[9];
cy q[12], q[3];
cx q[1], q[18];
ch q[17], q[9];
cy q[7], q[6];
cy q[17], q[4];
cx q[9], q[7];
cx q[22], q[16];
cu3(1.5707963267948966, 0, 0) q[17], q[22];
ch q[3], q[10];
cx q[13], q[11];
ch q[3], q[16];
ch q[1], q[8];
cy q[8], q[6];
cx q[9], q[13];
cy q[13], q[4];
cu3(1.5707963267948966, 0, 0) q[8], q[13];
cx q[11], q[16];
cx q[6], q[13];
cu3(1.5707963267948966, 0, 0) q[10], q[5];
ch q[15], q[7];
ch q[8], q[11];
ch q[6], q[15];
cx q[13], q[22];
cx q[15], q[19];
cy q[10], q[5];
cy q[6], q[7];
cx q[7], q[11];
cy q[10], q[16];
ch q[13], q[18];
cu3(1.5707963267948966, 0, 0) q[20], q[1];
cy q[12], q[6];
cy q[16], q[21];
cu3(1.5707963267948966, 0, 0) q[16], q[9];
cx q[15], q[17];
cx q[21], q[1];
cu3(1.5707963267948966, 0, 0) q[18], q[8];
cu3(1.5707963267948966, 0, 0) q[19], q[1];
ch q[10], q[20];
cu3(1.5707963267948966, 0, 0) q[18], q[11];
cx q[14], q[10];
