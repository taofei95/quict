OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
cx q[11], q[3];
cy q[0], q[7];
cu3(1.5707963267948966, 0, 0) q[3], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
ch q[9], q[7];
ch q[1], q[0];
cy q[6], q[1];
cu3(1.5707963267948966, 0, 0) q[11], q[8];
cy q[3], q[7];
ch q[11], q[9];
cu3(1.5707963267948966, 0, 0) q[3], q[8];
cu3(1.5707963267948966, 0, 0) q[6], q[5];
cx q[5], q[7];
cy q[6], q[8];
ch q[1], q[3];
cy q[6], q[4];
cu3(1.5707963267948966, 0, 0) q[3], q[10];
ch q[2], q[11];
cu3(1.5707963267948966, 0, 0) q[4], q[6];
cx q[11], q[1];
ch q[1], q[8];
cu3(1.5707963267948966, 0, 0) q[6], q[0];
ch q[7], q[9];
cu3(1.5707963267948966, 0, 0) q[2], q[11];
ch q[10], q[8];
cx q[2], q[5];
cy q[3], q[9];
cu3(1.5707963267948966, 0, 0) q[7], q[3];
cy q[9], q[8];
cu3(1.5707963267948966, 0, 0) q[6], q[0];
cy q[5], q[2];
ch q[2], q[10];
cx q[9], q[4];
cx q[9], q[2];
cx q[7], q[5];
cy q[8], q[9];
cx q[5], q[7];
cy q[4], q[11];
ch q[4], q[1];
cx q[3], q[8];
cy q[5], q[2];
ch q[6], q[4];
cu3(1.5707963267948966, 0, 0) q[9], q[11];
ch q[3], q[6];
cy q[8], q[1];
cy q[2], q[4];
ch q[9], q[4];
cx q[10], q[3];
cy q[3], q[1];
ch q[4], q[7];
cx q[8], q[9];
cu3(1.5707963267948966, 0, 0) q[9], q[1];
cy q[10], q[11];
cy q[7], q[5];
cx q[8], q[5];
ch q[2], q[7];
cu3(1.5707963267948966, 0, 0) q[3], q[5];
ch q[10], q[6];
ch q[0], q[4];
cy q[9], q[8];
ch q[4], q[3];
cy q[0], q[10];
cu3(1.5707963267948966, 0, 0) q[4], q[10];
cx q[3], q[1];
cx q[3], q[4];
ch q[6], q[8];
cx q[11], q[3];
cu3(1.5707963267948966, 0, 0) q[2], q[11];
cu3(1.5707963267948966, 0, 0) q[11], q[2];
cy q[0], q[2];
cx q[11], q[2];
ch q[11], q[3];
cu3(1.5707963267948966, 0, 0) q[5], q[0];
cu3(1.5707963267948966, 0, 0) q[4], q[10];
cx q[4], q[7];
ch q[2], q[5];
ch q[8], q[7];
cu3(1.5707963267948966, 0, 0) q[3], q[11];
cu3(1.5707963267948966, 0, 0) q[4], q[10];
cy q[10], q[5];
ch q[4], q[10];
cu3(1.5707963267948966, 0, 0) q[9], q[0];
cu3(1.5707963267948966, 0, 0) q[9], q[4];
cu3(1.5707963267948966, 0, 0) q[6], q[9];
