OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
ry(1.5707963267948966) q[0];
t q[0];
swap q[3], q[2];
tdg q[1];
tdg q[2];
p(0) q[3];
rz(1.5707963267948966) q[2];
h q[4];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[1];
cz q[2], q[3];
h q[1];
p(0) q[4];
s q[2];
cz q[3], q[1];
tdg q[0];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
t q[4];
id q[2];
h q[4];
id q[0];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[4];
cx q[2], q[1];
u1(1.5707963267948966) q[4];
t q[2];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
h q[0];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
swap q[4], q[1];
tdg q[4];
u1(1.5707963267948966) q[2];
cu1(1.5707963267948966) q[3], q[0];
cu1(1.5707963267948966) q[0], q[1];
p(0) q[4];
t q[2];
t q[1];
p(0) q[1];
rz(1.5707963267948966) q[0];
t q[3];
tdg q[4];
cu1(1.5707963267948966) q[4], q[3];
tdg q[3];
id q[3];
t q[4];
tdg q[3];
s q[0];
id q[3];
tdg q[1];
t q[3];
id q[1];
ch q[1], q[3];
u1(1.5707963267948966) q[0];
rxx(0) q[4], q[0];
rzz(1.5707963267948966) q[1], q[0];
rx(1.5707963267948966) q[3];
p(0) q[2];
cu3(1.5707963267948966, 0, 0) q[2], q[3];
u1(1.5707963267948966) q[2];