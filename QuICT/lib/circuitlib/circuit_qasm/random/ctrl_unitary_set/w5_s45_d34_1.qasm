OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
ch q[3], q[1];
cx q[3], q[0];
cx q[3], q[4];
cu3(1.5707963267948966, 0, 0) q[4], q[2];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
cy q[1], q[2];
cx q[0], q[2];
ch q[3], q[1];
cx q[3], q[1];
cy q[4], q[1];
cx q[4], q[1];
cu3(1.5707963267948966, 0, 0) q[3], q[4];
cy q[4], q[2];
cx q[4], q[1];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cy q[2], q[1];
cy q[2], q[0];
cx q[2], q[1];
cu3(1.5707963267948966, 0, 0) q[2], q[4];
cy q[0], q[3];
ch q[1], q[3];
ch q[1], q[0];
ch q[2], q[0];
cy q[3], q[0];
cx q[3], q[1];
cy q[3], q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[3];
cx q[4], q[0];
cy q[3], q[1];
cx q[1], q[2];
ch q[4], q[0];
cx q[4], q[0];
ch q[0], q[2];
cy q[3], q[1];
cx q[1], q[4];
cx q[0], q[2];
ch q[2], q[4];
cu3(1.5707963267948966, 0, 0) q[1], q[3];
cy q[1], q[2];
cx q[0], q[3];
cy q[0], q[1];
cx q[4], q[0];
cy q[3], q[2];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
cx q[0], q[2];
