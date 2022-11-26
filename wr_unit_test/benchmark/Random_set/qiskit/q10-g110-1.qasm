OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
u1(1.5707963267948966) q[0];
t q[7];
cz q[8], q[7];
tdg q[8];
u1(1.5707963267948966) q[6];
cz q[6], q[5];
rxx(0) q[3], q[5];
rx(1.5707963267948966) q[6];
p(0) q[2];
id q[0];
crz(1.5707963267948966) q[0], q[8];
rz(1.5707963267948966) q[1];
cu1(1.5707963267948966) q[1], q[6];
swap q[0], q[2];
u1(1.5707963267948966) q[8];
tdg q[8];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[0];
p(0) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
tdg q[9];
rzz(1.5707963267948966) q[6], q[5];
id q[9];
tdg q[6];
u1(1.5707963267948966) q[0];
id q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[5];
ch q[1], q[5];
cy q[6], q[8];
t q[0];
ry(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[4];
s q[8];
h q[0];
ry(1.5707963267948966) q[5];
cx q[6], q[1];
u1(1.5707963267948966) q[7];
tdg q[9];
u1(1.5707963267948966) q[4];
id q[8];
h q[7];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[1];
crz(1.5707963267948966) q[1], q[2];
t q[3];
rz(1.5707963267948966) q[2];
ch q[8], q[2];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[6];
swap q[3], q[1];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[9];
cu1(1.5707963267948966) q[4], q[2];
u1(1.5707963267948966) q[8];
p(0) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[4];
cu1(1.5707963267948966) q[7], q[3];
crz(1.5707963267948966) q[2], q[5];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[3];
p(0) q[7];
rx(1.5707963267948966) q[3];
sdg q[7];
id q[6];
tdg q[2];
p(0) q[2];
p(0) q[7];
u3(0, 0, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
swap q[8], q[5];
tdg q[2];
rx(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[0], q[9];
u1(1.5707963267948966) q[5];
tdg q[5];
id q[3];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[7];
cz q[7], q[5];
t q[2];
h q[3];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[3];
tdg q[0];
sdg q[2];
h q[1];
sdg q[7];
sdg q[9];
id q[9];
id q[8];
cz q[1], q[5];
rz(1.5707963267948966) q[6];
id q[2];
sdg q[9];
h q[6];
tdg q[1];
rz(1.5707963267948966) q[7];
t q[4];
u1(1.5707963267948966) q[4];
ch q[0], q[1];
u3(0, 0, 1.5707963267948966) q[8];
crz(1.5707963267948966) q[1], q[7];
tdg q[0];