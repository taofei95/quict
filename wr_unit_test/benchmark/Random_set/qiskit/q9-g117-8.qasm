OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
p(0) q[1];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[2];
id q[2];
u1(1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[8];
rx(1.5707963267948966) q[4];
p(0) q[4];
p(0) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rxx(0) q[8], q[6];
u1(1.5707963267948966) q[8];
t q[5];
t q[4];
tdg q[4];
cu1(1.5707963267948966) q[3], q[8];
sdg q[2];
rzz(1.5707963267948966) q[0], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
t q[6];
rx(1.5707963267948966) q[1];
cy q[0], q[2];
cy q[5], q[6];
u1(1.5707963267948966) q[4];
s q[5];
cu1(1.5707963267948966) q[6], q[2];
cx q[6], q[1];
ch q[0], q[8];
crz(1.5707963267948966) q[8], q[6];
ch q[5], q[3];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[5];
s q[2];
cu1(1.5707963267948966) q[1], q[7];
crz(1.5707963267948966) q[1], q[5];
cy q[5], q[6];
swap q[0], q[3];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
id q[0];
sdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[6];
sdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[8];
id q[7];
crz(1.5707963267948966) q[5], q[4];
t q[4];
t q[3];
h q[6];
s q[6];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[6];
cy q[3], q[8];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
cx q[5], q[7];
h q[3];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[3];
crz(1.5707963267948966) q[6], q[3];
id q[6];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cz q[1], q[3];
tdg q[2];
s q[2];
id q[3];
t q[2];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[5];
rz(1.5707963267948966) q[4];
cy q[7], q[3];
u1(1.5707963267948966) q[0];
sdg q[0];
h q[0];
tdg q[4];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[0];
sdg q[6];
u1(1.5707963267948966) q[3];
s q[7];
rxx(0) q[1], q[7];
tdg q[5];
u3(0, 0, 1.5707963267948966) q[4];
p(0) q[3];
id q[5];
u3(0, 0, 1.5707963267948966) q[1];
cu1(1.5707963267948966) q[4], q[5];
u3(0, 0, 1.5707963267948966) q[6];
s q[7];
h q[6];
tdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[8];
cu3(1.5707963267948966, 0, 0) q[3], q[6];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[3];
id q[6];
sdg q[4];
s q[8];
sdg q[0];
swap q[5], q[4];
id q[5];
rz(1.5707963267948966) q[8];
crz(1.5707963267948966) q[2], q[0];
u3(0, 0, 1.5707963267948966) q[8];
id q[8];