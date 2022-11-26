OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
sdg q[4];
sdg q[4];
cu3(1.5707963267948966, 0, 0) q[1], q[5];
ry(1.5707963267948966) q[5];
rxx(0) q[2], q[0];
s q[3];
h q[4];
p(0) q[0];
t q[1];
u1(1.5707963267948966) q[5];
h q[4];
rzz(1.5707963267948966) q[0], q[1];
ry(1.5707963267948966) q[2];
s q[3];
tdg q[0];
t q[3];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
p(0) q[1];
sdg q[2];
rx(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[4], q[3];
h q[4];
s q[5];
rzz(1.5707963267948966) q[4], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
rzz(1.5707963267948966) q[3], q[1];
u1(1.5707963267948966) q[0];
cu1(1.5707963267948966) q[0], q[2];
u1(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
id q[5];
u1(1.5707963267948966) q[5];
id q[0];
rx(1.5707963267948966) q[1];
p(0) q[2];
cy q[1], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
h q[1];
tdg q[1];
rz(1.5707963267948966) q[0];
rxx(0) q[3], q[0];
p(0) q[2];
sdg q[0];
s q[1];
cy q[3], q[4];
t q[3];
t q[4];
p(0) q[0];
s q[3];
ch q[4], q[5];
h q[5];
tdg q[5];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[3];
rx(1.5707963267948966) q[5];
h q[3];
h q[3];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[3];
tdg q[3];
t q[5];
rzz(1.5707963267948966) q[1], q[5];
t q[1];
swap q[5], q[1];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
sdg q[1];
h q[4];
rx(1.5707963267948966) q[4];
sdg q[1];
cu3(1.5707963267948966, 0, 0) q[4], q[1];
sdg q[0];
ry(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[0], q[5];
rz(1.5707963267948966) q[0];
swap q[0], q[1];
rx(1.5707963267948966) q[5];
tdg q[2];
tdg q[3];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[1];
s q[4];
ry(1.5707963267948966) q[4];
p(0) q[5];
u3(0, 0, 1.5707963267948966) q[5];
id q[0];