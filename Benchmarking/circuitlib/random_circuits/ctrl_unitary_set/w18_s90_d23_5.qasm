OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
cy q[10], q[7];
cx q[11], q[1];
cu3(1.5707963267948966, 0, 0) q[3], q[13];
ch q[8], q[12];
cu3(1.5707963267948966, 0, 0) q[17], q[11];
cu3(1.5707963267948966, 0, 0) q[1], q[14];
cx q[7], q[15];
cu3(1.5707963267948966, 0, 0) q[2], q[6];
cu3(1.5707963267948966, 0, 0) q[3], q[7];
cx q[5], q[1];
cu3(1.5707963267948966, 0, 0) q[5], q[0];
cx q[10], q[6];
ch q[7], q[16];
ch q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[17], q[1];
cx q[13], q[0];
ch q[0], q[17];
cu3(1.5707963267948966, 0, 0) q[13], q[1];
cy q[14], q[12];
cy q[8], q[3];
cy q[11], q[8];
cu3(1.5707963267948966, 0, 0) q[13], q[7];
ch q[0], q[2];
ch q[17], q[16];
cu3(1.5707963267948966, 0, 0) q[6], q[11];
cu3(1.5707963267948966, 0, 0) q[5], q[8];
cx q[8], q[1];
ch q[14], q[10];
cx q[4], q[3];
cx q[13], q[6];
ch q[2], q[17];
cu3(1.5707963267948966, 0, 0) q[16], q[17];
ch q[10], q[7];
cu3(1.5707963267948966, 0, 0) q[14], q[10];
cx q[0], q[5];
cy q[3], q[13];
cx q[7], q[4];
cu3(1.5707963267948966, 0, 0) q[13], q[9];
cu3(1.5707963267948966, 0, 0) q[5], q[1];
cy q[10], q[7];
ch q[10], q[15];
cu3(1.5707963267948966, 0, 0) q[17], q[2];
cu3(1.5707963267948966, 0, 0) q[4], q[13];
cy q[13], q[10];
cx q[14], q[4];
cu3(1.5707963267948966, 0, 0) q[16], q[0];
ch q[3], q[15];
cu3(1.5707963267948966, 0, 0) q[16], q[4];
cx q[5], q[6];
ch q[17], q[11];
cy q[15], q[8];
cx q[3], q[14];
cx q[9], q[11];
cy q[7], q[13];
cu3(1.5707963267948966, 0, 0) q[6], q[11];
cy q[10], q[13];
cx q[9], q[0];
ch q[9], q[13];
cx q[6], q[5];
cy q[1], q[17];
cu3(1.5707963267948966, 0, 0) q[5], q[11];
cx q[17], q[4];
cx q[13], q[10];
ch q[3], q[8];
cy q[3], q[9];
cy q[9], q[3];
ch q[12], q[14];
cx q[2], q[6];
cx q[17], q[12];
ch q[2], q[0];
ch q[17], q[11];
ch q[4], q[2];
cu3(1.5707963267948966, 0, 0) q[7], q[17];
cx q[15], q[1];
cx q[2], q[16];
ch q[6], q[13];
cy q[15], q[3];
ch q[11], q[17];
ch q[0], q[12];
ch q[3], q[11];
ch q[9], q[10];
ch q[11], q[16];
cy q[17], q[8];
cx q[17], q[15];
cy q[9], q[2];
cu3(1.5707963267948966, 0, 0) q[15], q[8];
cx q[8], q[12];
cx q[0], q[15];
ch q[5], q[17];
cu3(1.5707963267948966, 0, 0) q[14], q[11];