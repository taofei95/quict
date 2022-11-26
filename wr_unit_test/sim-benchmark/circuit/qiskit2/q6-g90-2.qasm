OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
cu1(1.5707963267948966) q[5], q[1];
sdg q[5];
u1(1.5707963267948966) q[3];
tdg q[2];
s q[2];
s q[3];
rz(1.5707963267948966) q[3];
rzz(1.5707963267948966) q[1], q[2];
p(0) q[2];
cx q[5], q[0];
rx(1.5707963267948966) q[2];
p(0) q[1];
rxx(0) q[1], q[3];
u1(1.5707963267948966) q[0];
cy q[2], q[3];
tdg q[5];
h q[4];
cy q[0], q[3];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
tdg q[0];
cu1(1.5707963267948966) q[4], q[5];
cy q[4], q[5];
sdg q[2];
sdg q[5];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[5];
rx(1.5707963267948966) q[5];
swap q[5], q[3];
id q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
p(0) q[3];
ry(1.5707963267948966) q[5];
id q[4];
id q[1];
ry(1.5707963267948966) q[3];
p(0) q[3];
rxx(0) q[1], q[2];
ry(1.5707963267948966) q[5];
rxx(0) q[0], q[2];
u3(0, 0, 1.5707963267948966) q[0];
swap q[1], q[3];
ry(1.5707963267948966) q[3];
rzz(1.5707963267948966) q[0], q[2];
rz(1.5707963267948966) q[2];
id q[1];
cx q[5], q[2];
rz(1.5707963267948966) q[3];
h q[5];
tdg q[0];
cz q[4], q[2];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
h q[2];
h q[5];
rx(1.5707963267948966) q[4];
s q[5];
ry(1.5707963267948966) q[0];
p(0) q[0];
id q[2];
u1(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
s q[4];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
t q[2];
cx q[5], q[4];
rzz(1.5707963267948966) q[1], q[2];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
t q[3];
swap q[5], q[4];
sdg q[4];
cz q[3], q[4];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
s q[3];
sdg q[1];
p(0) q[1];
s q[5];
t q[3];
u1(1.5707963267948966) q[1];