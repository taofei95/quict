OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
p(0) q[2];
s q[2];
h q[3];
u3(0, 0, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[1];
p(0) q[0];
h q[3];
id q[3];
u3(0, 0, 1.5707963267948966) q[2];
s q[3];
u3(0, 0, 1.5707963267948966) q[3];
tdg q[2];
ch q[3], q[0];
h q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
rxx(0) q[4], q[0];
swap q[3], q[1];
p(0) q[3];
u1(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
id q[4];
ry(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
id q[4];
rz(1.5707963267948966) q[4];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[4];
tdg q[2];
cy q[1], q[4];
p(0) q[0];
tdg q[2];
p(0) q[0];
cx q[1], q[2];
id q[0];
s q[3];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
h q[0];
ry(1.5707963267948966) q[4];
swap q[4], q[1];