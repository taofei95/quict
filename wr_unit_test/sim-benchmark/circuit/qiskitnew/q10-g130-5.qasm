OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
sdg q[4];
u1(1.5707963267948966) q[9];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[9];
id q[9];
t q[5];
s q[7];
id q[0];
u1(1.5707963267948966) q[8];
sdg q[2];
p(0) q[6];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
h q[7];
rz(1.5707963267948966) q[3];
p(0) q[7];
t q[7];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[2];
swap q[3], q[1];
rz(1.5707963267948966) q[2];
id q[1];
rz(1.5707963267948966) q[1];
cx q[0], q[6];
u3(0, 0, 1.5707963267948966) q[8];
p(0) q[3];
t q[5];
sdg q[9];
rz(1.5707963267948966) q[4];
sdg q[4];
s q[5];
h q[1];
cy q[6], q[0];
u1(1.5707963267948966) q[7];
h q[4];
id q[4];
id q[3];
tdg q[1];
p(0) q[5];
u1(1.5707963267948966) q[4];
tdg q[2];
p(0) q[8];
s q[8];
id q[3];
sdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[8];
sdg q[3];
u3(0, 0, 1.5707963267948966) q[4];
p(0) q[2];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[0];
sdg q[3];
swap q[4], q[1];
sdg q[5];
cu3(1.5707963267948966, 0, 0) q[1], q[9];
cu3(1.5707963267948966, 0, 0) q[3], q[1];
rx(1.5707963267948966) q[5];
h q[2];
u1(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[0];
h q[2];
swap q[0], q[3];
cz q[8], q[3];
s q[6];
u1(1.5707963267948966) q[2];
cx q[0], q[3];
t q[7];
sdg q[2];
p(0) q[5];
s q[6];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[7];
s q[5];
ry(1.5707963267948966) q[5];
h q[7];
t q[5];
cy q[6], q[5];
s q[5];
ry(1.5707963267948966) q[1];
h q[3];
s q[4];
cx q[8], q[7];
p(0) q[5];
s q[2];
ry(1.5707963267948966) q[5];
cx q[0], q[9];
sdg q[5];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[1];
cy q[5], q[9];
rzz(1.5707963267948966) q[2], q[6];
cu1(1.5707963267948966) q[3], q[1];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
p(0) q[1];
cy q[8], q[1];
swap q[8], q[5];
h q[0];
cz q[2], q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[9];
h q[2];
tdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
h q[7];
sdg q[3];
sdg q[4];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[8];
u1(1.5707963267948966) q[3];
cu1(1.5707963267948966) q[0], q[3];
tdg q[3];
cx q[8], q[6];
p(0) q[3];
rx(1.5707963267948966) q[9];
t q[8];
u2(1.5707963267948966, 1.5707963267948966) q[7];
cz q[3], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
p(0) q[8];
rzz(1.5707963267948966) q[8], q[7];
t q[2];
rz(1.5707963267948966) q[1];