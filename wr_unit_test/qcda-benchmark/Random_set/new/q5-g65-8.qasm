OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
crz(1.5707963267948966) q[0], q[1];
u1(1.5707963267948966) q[4];
x q[3];
ry(1.5707963267948966) q[3];
t q[4];
rz(1.5707963267948966) q[0];
s q[4];
rz(1.5707963267948966) q[4];
p(0) q[0];
tdg q[4];
rx(1.5707963267948966) q[3];
t q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rxx(0) q[0], q[4];
id q[3];
u1(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
swap q[0], q[4];
rz(1.5707963267948966) q[2];
id q[0];
tdg q[1];
s q[4];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
sdg q[3];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[4];
ry(1.5707963267948966) q[4];
cx q[2], q[0];
crz(1.5707963267948966) q[4], q[1];
swap q[0], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
s q[1];
p(0) q[1];
tdg q[1];
rz(1.5707963267948966) q[4];
rxx(0) q[1], q[3];
u1(1.5707963267948966) q[0];
p(0) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
t q[4];
t q[1];
cu1(1.5707963267948966) q[4], q[0];
rx(1.5707963267948966) q[2];
t q[4];
h q[2];
u1(1.5707963267948966) q[0];
id q[2];
s q[2];
sdg q[3];
tdg q[3];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[0];
id q[1];
u1(1.5707963267948966) q[4];
s q[2];
t q[4];
sdg q[0];
ry(1.5707963267948966) q[2];
cx q[3], q[1];
u1(1.5707963267948966) q[4];
t q[4];