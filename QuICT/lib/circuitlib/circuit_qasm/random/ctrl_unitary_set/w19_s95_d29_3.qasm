OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
ch q[12], q[8];
cx q[16], q[15];
cx q[7], q[17];
ch q[4], q[10];
cy q[17], q[7];
ch q[15], q[4];
cx q[0], q[1];
cx q[8], q[16];
cy q[6], q[12];
ch q[11], q[12];
cu3(1.5707963267948966, 0, 0) q[6], q[15];
cu3(1.5707963267948966, 0, 0) q[13], q[4];
ch q[4], q[1];
ch q[18], q[17];
ch q[1], q[14];
ch q[6], q[3];
cy q[8], q[12];
ch q[6], q[12];
cy q[1], q[12];
ch q[8], q[14];
cy q[5], q[9];
cx q[1], q[17];
cy q[2], q[18];
cy q[10], q[15];
cy q[4], q[16];
cx q[13], q[10];
cx q[3], q[0];
cy q[2], q[3];
cu3(1.5707963267948966, 0, 0) q[2], q[8];
cy q[11], q[9];
ch q[7], q[5];
cy q[10], q[13];
cy q[0], q[13];
ch q[14], q[7];
cy q[12], q[18];
cx q[0], q[9];
cx q[7], q[3];
ch q[14], q[9];
ch q[11], q[9];
cu3(1.5707963267948966, 0, 0) q[1], q[7];
cx q[13], q[1];
cx q[13], q[5];
cy q[5], q[1];
cy q[2], q[5];
cy q[12], q[11];
cy q[1], q[9];
cy q[17], q[11];
cy q[2], q[4];
cu3(1.5707963267948966, 0, 0) q[13], q[16];
cx q[12], q[1];
cy q[8], q[15];
ch q[9], q[18];
ch q[7], q[11];
cu3(1.5707963267948966, 0, 0) q[15], q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[12];
ch q[1], q[17];
ch q[14], q[0];
cy q[12], q[6];
cx q[11], q[1];
cx q[13], q[7];
ch q[16], q[7];
cy q[14], q[0];
ch q[1], q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
cy q[1], q[0];
ch q[3], q[17];
cx q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[9], q[14];
cx q[9], q[4];
cx q[0], q[14];
ch q[10], q[9];
cu3(1.5707963267948966, 0, 0) q[8], q[1];
cu3(1.5707963267948966, 0, 0) q[10], q[18];
ch q[3], q[1];
ch q[13], q[14];
cy q[13], q[10];
cx q[8], q[14];
ch q[13], q[12];
cx q[13], q[9];
ch q[6], q[9];
cy q[15], q[9];
cx q[4], q[12];
cu3(1.5707963267948966, 0, 0) q[17], q[10];
cu3(1.5707963267948966, 0, 0) q[16], q[5];
ch q[10], q[8];
cx q[0], q[10];
cu3(1.5707963267948966, 0, 0) q[3], q[4];
cx q[4], q[15];
cx q[9], q[5];
cu3(1.5707963267948966, 0, 0) q[1], q[13];
cx q[3], q[10];
cu3(1.5707963267948966, 0, 0) q[2], q[16];
cu3(1.5707963267948966, 0, 0) q[15], q[4];
ch q[2], q[1];
cy q[3], q[17];
