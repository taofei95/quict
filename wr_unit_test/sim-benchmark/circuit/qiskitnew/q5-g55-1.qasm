OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
tdg q[1];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[1];
t q[2];
rx(1.5707963267948966) q[0];
cy q[1], q[2];
p(0) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
t q[0];
rx(1.5707963267948966) q[4];
tdg q[0];
t q[3];
cy q[1], q[2];
p(0) q[3];
p(0) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
sdg q[0];
t q[2];
u1(1.5707963267948966) q[1];
t q[2];
rxx(0) q[3], q[1];
rz(1.5707963267948966) q[4];
id q[2];
p(0) q[1];
h q[1];
h q[1];
tdg q[3];
id q[4];
id q[3];
cx q[3], q[0];
h q[0];
id q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rxx(0) q[3], q[4];
t q[0];
t q[3];
h q[1];
t q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
id q[0];
ry(1.5707963267948966) q[2];
sdg q[0];
h q[2];
s q[1];
rz(1.5707963267948966) q[0];
s q[2];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[4];
id q[3];
rz(1.5707963267948966) q[4];