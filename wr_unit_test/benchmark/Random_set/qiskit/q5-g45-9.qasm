OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
t q[3];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
t q[3];
t q[0];
ch q[0], q[2];
id q[3];
h q[1];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[1];
t q[0];
rx(1.5707963267948966) q[0];
swap q[3], q[4];
rx(1.5707963267948966) q[4];
tdg q[2];
t q[2];
rxx(0) q[2], q[0];
cx q[0], q[3];
p(0) q[2];
id q[0];
ry(1.5707963267948966) q[3];
h q[3];
u3(0, 0, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[4];
id q[0];
id q[0];
sdg q[2];
cu1(1.5707963267948966) q[1], q[2];
sdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
tdg q[4];
t q[3];
s q[4];
h q[3];
swap q[2], q[0];
p(0) q[4];
crz(1.5707963267948966) q[4], q[1];
cy q[4], q[3];
rz(1.5707963267948966) q[0];
p(0) q[2];
rz(1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
ry(1.5707963267948966) q[0];