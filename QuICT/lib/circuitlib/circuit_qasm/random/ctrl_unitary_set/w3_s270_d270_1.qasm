OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
cx q[0], q[2];
cx q[2], q[0];
cy q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
ch q[2], q[0];
ch q[1], q[0];
ch q[1], q[2];
ch q[2], q[1];
ch q[1], q[0];
cy q[2], q[1];
cy q[1], q[0];
cy q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
cx q[1], q[2];
ch q[1], q[2];
ch q[0], q[2];
ch q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
cx q[1], q[0];
cx q[1], q[0];
cy q[0], q[1];
ch q[2], q[0];
cx q[0], q[2];
cx q[0], q[2];
cx q[1], q[0];
cy q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
cx q[0], q[2];
cx q[0], q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
cx q[2], q[0];
cx q[1], q[2];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
cx q[2], q[1];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
ch q[0], q[1];
cy q[1], q[0];
cy q[1], q[2];
cx q[1], q[2];
cx q[2], q[0];
cx q[2], q[1];
cx q[1], q[2];
ch q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cy q[1], q[2];
cx q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
ch q[2], q[1];
ch q[1], q[0];
ch q[1], q[2];
cy q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
cx q[0], q[2];
cx q[1], q[2];
cx q[0], q[2];
cx q[1], q[0];
cy q[2], q[0];
cy q[1], q[2];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
cy q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cy q[1], q[2];
ch q[0], q[1];
cx q[2], q[1];
cx q[1], q[0];
cy q[0], q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
cy q[0], q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
cx q[2], q[0];
cx q[1], q[2];
ch q[2], q[0];
cx q[1], q[2];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
ch q[1], q[0];
cx q[0], q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cx q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cy q[0], q[2];
ch q[0], q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cx q[0], q[2];
ch q[1], q[0];
cy q[1], q[2];
cx q[0], q[2];
ch q[2], q[0];
cx q[0], q[1];
cy q[0], q[2];
ch q[0], q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
ch q[1], q[0];
cx q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
cx q[1], q[2];
ch q[1], q[0];
cx q[2], q[0];
cy q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
ch q[1], q[0];
cx q[1], q[2];
ch q[2], q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
ch q[0], q[2];
ch q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
cx q[2], q[0];
cy q[0], q[1];
ch q[1], q[2];
cx q[1], q[2];
cx q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
ch q[0], q[1];
ch q[0], q[2];
cy q[1], q[2];
ch q[1], q[0];
cy q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cx q[0], q[1];
cy q[1], q[0];
cx q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cy q[1], q[0];
cx q[1], q[2];
cy q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
ch q[2], q[0];
ch q[0], q[2];
ch q[1], q[2];
cx q[0], q[1];
cy q[2], q[0];
cx q[1], q[0];
cy q[2], q[0];
cx q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
cy q[0], q[2];
cx q[0], q[2];
cx q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
cx q[2], q[1];
cx q[2], q[1];
cx q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
ch q[2], q[1];
ch q[0], q[1];
ch q[0], q[1];
cy q[2], q[0];
ch q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
ch q[2], q[0];
ch q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cx q[0], q[2];
ch q[1], q[0];
ch q[0], q[2];
ch q[2], q[1];
cy q[2], q[0];
ch q[1], q[2];
cy q[2], q[0];
cx q[2], q[0];
ch q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
cy q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cx q[0], q[1];
cx q[0], q[1];
cx q[1], q[0];
ch q[0], q[1];
ch q[1], q[0];
ch q[0], q[1];
cx q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
ch q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cy q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
ch q[0], q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
cx q[2], q[0];
ch q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cy q[1], q[2];
cx q[2], q[0];
cx q[2], q[0];
ch q[1], q[2];
cy q[0], q[1];
ch q[1], q[0];
ch q[1], q[0];
cx q[1], q[2];
cx q[2], q[1];
cx q[1], q[2];
cx q[2], q[0];
cx q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
ch q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
cx q[2], q[1];
cx q[2], q[0];
ch q[2], q[1];
cx q[1], q[2];
cy q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
ch q[0], q[1];
ch q[1], q[2];
cy q[0], q[1];
cy q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
cx q[2], q[1];
cy q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
cy q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
cx q[1], q[2];
cy q[1], q[0];
ch q[2], q[0];
cx q[1], q[2];
cx q[2], q[1];
ch q[1], q[2];
cy q[0], q[1];
cx q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
ch q[2], q[1];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
cy q[0], q[2];
cy q[0], q[2];
cy q[1], q[2];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
cx q[2], q[1];
cx q[1], q[2];
cx q[1], q[0];
ch q[0], q[2];
cx q[2], q[1];
cy q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
ch q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cy q[0], q[1];
cy q[2], q[0];
ch q[0], q[2];
cx q[1], q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
cx q[0], q[2];
ch q[1], q[2];
ch q[2], q[0];
ch q[0], q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
cx q[1], q[2];
ch q[0], q[2];
cx q[1], q[2];
cy q[1], q[2];
