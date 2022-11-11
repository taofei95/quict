OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
cx q[8], q[2];
cu3(1.5707963267948966, 0, 0) q[20], q[13];
cy q[16], q[13];
ch q[20], q[14];
cx q[8], q[7];
ch q[20], q[3];
cx q[7], q[16];
cu3(1.5707963267948966, 0, 0) q[14], q[3];
cy q[16], q[5];
ch q[19], q[17];
cx q[4], q[11];
ch q[7], q[20];
ch q[12], q[9];
ch q[22], q[10];
cy q[14], q[4];
cy q[11], q[21];
cy q[18], q[5];
cx q[21], q[16];
ch q[18], q[5];
cu3(1.5707963267948966, 0, 0) q[6], q[16];
cy q[13], q[19];
cu3(1.5707963267948966, 0, 0) q[10], q[11];
cu3(1.5707963267948966, 0, 0) q[19], q[11];
ch q[13], q[15];
ch q[9], q[15];
cy q[19], q[16];
cx q[6], q[19];
ch q[22], q[9];
cy q[17], q[19];
cy q[0], q[21];
ch q[19], q[17];
cu3(1.5707963267948966, 0, 0) q[6], q[12];
cu3(1.5707963267948966, 0, 0) q[15], q[3];
cy q[18], q[19];
ch q[9], q[16];
cx q[3], q[17];
cu3(1.5707963267948966, 0, 0) q[1], q[9];
cx q[1], q[14];
cu3(1.5707963267948966, 0, 0) q[22], q[11];
cx q[9], q[11];
ch q[22], q[21];
cu3(1.5707963267948966, 0, 0) q[12], q[19];
cx q[22], q[3];
cx q[3], q[1];
ch q[19], q[3];
cy q[14], q[17];
cy q[4], q[14];
ch q[6], q[13];
cy q[22], q[8];
cy q[5], q[4];
cu3(1.5707963267948966, 0, 0) q[22], q[10];
cy q[2], q[11];
ch q[5], q[2];
cx q[22], q[6];
ch q[2], q[13];
cy q[14], q[2];
cu3(1.5707963267948966, 0, 0) q[1], q[16];
ch q[20], q[12];
cy q[7], q[3];
cx q[6], q[13];
cy q[20], q[7];
ch q[13], q[17];
cx q[14], q[22];
cx q[11], q[10];
cy q[10], q[13];
cy q[21], q[22];
cu3(1.5707963267948966, 0, 0) q[18], q[19];
cy q[21], q[6];
cy q[1], q[15];
cy q[21], q[5];
cu3(1.5707963267948966, 0, 0) q[22], q[19];
cy q[8], q[12];
cu3(1.5707963267948966, 0, 0) q[15], q[17];
cu3(1.5707963267948966, 0, 0) q[19], q[3];
cy q[15], q[4];
cx q[4], q[0];
cy q[20], q[11];
ch q[13], q[5];
cx q[9], q[12];
cu3(1.5707963267948966, 0, 0) q[11], q[20];
cx q[20], q[11];
ch q[9], q[2];
cx q[16], q[18];
cx q[13], q[12];
cx q[3], q[11];
ch q[3], q[10];
cy q[5], q[9];
cy q[17], q[22];
cy q[3], q[11];
cu3(1.5707963267948966, 0, 0) q[13], q[7];
cx q[4], q[9];
cu3(1.5707963267948966, 0, 0) q[11], q[22];
ch q[20], q[5];
cu3(1.5707963267948966, 0, 0) q[1], q[8];
cy q[10], q[19];
cy q[22], q[5];
cx q[22], q[17];
cy q[10], q[12];
cu3(1.5707963267948966, 0, 0) q[2], q[7];
cu3(1.5707963267948966, 0, 0) q[3], q[6];
cy q[8], q[21];
cy q[7], q[4];
cx q[18], q[20];
ch q[14], q[5];
cx q[15], q[17];
cy q[16], q[12];
cx q[5], q[7];
cx q[13], q[17];
cy q[6], q[19];
cu3(1.5707963267948966, 0, 0) q[12], q[9];
cu3(1.5707963267948966, 0, 0) q[15], q[11];
cy q[4], q[22];
cu3(1.5707963267948966, 0, 0) q[8], q[0];
cu3(1.5707963267948966, 0, 0) q[12], q[9];
ch q[6], q[0];
cy q[1], q[2];
cy q[9], q[8];
ch q[6], q[21];
cu3(1.5707963267948966, 0, 0) q[12], q[13];
cu3(1.5707963267948966, 0, 0) q[5], q[13];
cx q[2], q[19];
cx q[7], q[22];
ch q[3], q[2];
ch q[8], q[20];
cx q[10], q[1];
cy q[9], q[6];
cy q[15], q[0];
cx q[9], q[19];
cu3(1.5707963267948966, 0, 0) q[0], q[14];
ch q[19], q[12];
cy q[11], q[14];
cx q[3], q[14];
cy q[4], q[22];
cy q[14], q[18];
cy q[1], q[4];
cx q[17], q[6];
cy q[17], q[0];
cy q[3], q[14];
