OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
u2(1.5707963267948966, 1.5707963267948966) q[2];
p(0) q[1];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
ch q[0], q[5];
h q[3];
cu3(1.5707963267948966, 0, 0) q[5], q[0];
p(0) q[0];
id q[3];
swap q[0], q[4];
p(0) q[3];
s q[4];
u3(0, 0, 1.5707963267948966) q[3];
rxx(0) q[4], q[5];
cy q[5], q[3];
s q[3];
p(0) q[0];
cx q[5], q[4];
p(0) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
cx q[4], q[5];
tdg q[4];
rz(1.5707963267948966) q[1];
sdg q[1];
rz(1.5707963267948966) q[1];
p(0) q[5];
t q[1];