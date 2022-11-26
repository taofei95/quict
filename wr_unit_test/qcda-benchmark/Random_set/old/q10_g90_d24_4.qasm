OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
s q[7];
rxx(0) q[5], q[4];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[3];
cu1(1.5707963267948966) q[8], q[4];
h q[4];
cx q[2], q[6];
id q[2];
swap q[9], q[2];
u1(1.5707963267948966) q[0];
tdg q[2];
p(0) q[5];
rx(1.5707963267948966) q[6];
t q[7];
h q[8];
u2(1.5707963267948966, 1.5707963267948966) q[8];
sdg q[4];
h q[7];
t q[9];
h q[6];
x q[4];
u1(1.5707963267948966) q[3];
tdg q[8];
t q[3];
t q[2];
swap q[6], q[4];
u1(1.5707963267948966) q[4];
x q[4];
cu3(1.5707963267948966, 0, 0) q[4], q[1];
p(0) q[1];
h q[8];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[4];
tdg q[4];
t q[0];
cu1(1.5707963267948966) q[9], q[2];
tdg q[5];
u1(1.5707963267948966) q[1];
ch q[4], q[1];
ry(1.5707963267948966) q[1];
t q[4];
cu3(1.5707963267948966, 0, 0) q[9], q[6];
sdg q[4];
rz(1.5707963267948966) q[5];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rzz(1.5707963267948966) q[4], q[8];
p(0) q[4];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[8];
rxx(0) q[9], q[5];
rzz(1.5707963267948966) q[6], q[2];
tdg q[2];
cx q[5], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[6];
rzz(1.5707963267948966) q[8], q[2];
u3(0, 0, 1.5707963267948966) q[9];
h q[6];
cu1(1.5707963267948966) q[7], q[3];
id q[6];
s q[9];
tdg q[5];
p(0) q[8];
ry(1.5707963267948966) q[8];
s q[7];
sdg q[3];
tdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[7];
ch q[1], q[2];
cx q[7], q[9];
u3(0, 0, 1.5707963267948966) q[5];
ryy(1.5707963267948966) q[9], q[3];
cy q[4], q[6];
u1(1.5707963267948966) q[4];
crz(1.5707963267948966) q[6], q[9];
x q[0];
x q[9];
h q[7];
t q[6];
cy q[8], q[6];
cz q[5], q[9];
u1(1.5707963267948966) q[5];
t q[5];
p(0) q[7];