OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
rzz(1.5707963267948966) q[5], q[0];
rzz(1.5707963267948966) q[2], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[4];
tdg q[6];
h q[3];
rxx(0) q[1], q[7];
h q[3];
u1(1.5707963267948966) q[3];
cy q[4], q[2];
rz(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[4], q[0];
p(0) q[1];
rz(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[5];
cx q[3], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
cy q[5], q[0];
rx(1.5707963267948966) q[0];
id q[5];
rz(1.5707963267948966) q[2];
h q[0];
ry(1.5707963267948966) q[4];
t q[3];
u1(1.5707963267948966) q[1];
swap q[1], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rzz(1.5707963267948966) q[6], q[3];
ch q[1], q[4];
rz(1.5707963267948966) q[0];
p(0) q[0];
rx(1.5707963267948966) q[2];
id q[0];
id q[6];
id q[0];
cz q[4], q[3];
p(0) q[5];