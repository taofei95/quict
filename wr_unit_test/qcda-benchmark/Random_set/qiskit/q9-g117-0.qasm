OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
rx(1.5707963267948966) q[0];
rzz(1.5707963267948966) q[3], q[8];
tdg q[2];
p(0) q[4];
tdg q[5];
sdg q[1];
p(0) q[5];
t q[3];
u1(1.5707963267948966) q[6];
crz(1.5707963267948966) q[8], q[6];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[6];
cx q[1], q[6];
tdg q[1];
h q[7];
crz(1.5707963267948966) q[1], q[0];
cu3(1.5707963267948966, 0, 0) q[6], q[5];
id q[4];
sdg q[8];
cz q[0], q[2];
sdg q[7];
t q[2];
t q[6];
sdg q[1];
id q[4];
rz(1.5707963267948966) q[6];
p(0) q[0];
u1(1.5707963267948966) q[6];
cu1(1.5707963267948966) q[2], q[7];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[6];
s q[3];
h q[2];
sdg q[3];
tdg q[3];
u1(1.5707963267948966) q[1];
cu1(1.5707963267948966) q[8], q[4];
rx(1.5707963267948966) q[8];
swap q[1], q[3];
s q[5];
swap q[3], q[2];
cy q[3], q[8];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[3];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[4];
crz(1.5707963267948966) q[6], q[7];
t q[1];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[7];
u1(1.5707963267948966) q[6];
p(0) q[2];
sdg q[4];
t q[2];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[0];
tdg q[3];
tdg q[8];
h q[6];
ry(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
ch q[5], q[4];
h q[8];
tdg q[0];
cu1(1.5707963267948966) q[1], q[3];
u1(1.5707963267948966) q[8];
id q[0];
swap q[1], q[7];
h q[2];
u1(1.5707963267948966) q[6];
crz(1.5707963267948966) q[5], q[3];
p(0) q[6];
s q[5];
s q[4];
u3(0, 0, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[1];
ch q[6], q[8];
u3(0, 0, 1.5707963267948966) q[0];
s q[1];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
id q[8];
ry(1.5707963267948966) q[3];
tdg q[4];
id q[5];
sdg q[4];
u1(1.5707963267948966) q[6];
rzz(1.5707963267948966) q[2], q[4];
p(0) q[1];
swap q[3], q[8];
u3(0, 0, 1.5707963267948966) q[2];
t q[5];
tdg q[0];
swap q[1], q[0];
rz(1.5707963267948966) q[2];
sdg q[0];
cx q[7], q[6];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[7];
ry(1.5707963267948966) q[7];
id q[2];
id q[2];
ry(1.5707963267948966) q[0];
sdg q[0];
u3(0, 0, 1.5707963267948966) q[7];
s q[3];
t q[0];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
p(0) q[8];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[8];