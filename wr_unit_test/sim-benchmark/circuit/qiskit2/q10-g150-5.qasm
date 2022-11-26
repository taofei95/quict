OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
cz q[4], q[5];
tdg q[4];
sdg q[7];
tdg q[3];
swap q[5], q[4];
u3(0, 0, 1.5707963267948966) q[5];
s q[8];
rzz(1.5707963267948966) q[9], q[6];
cy q[5], q[1];
t q[4];
rzz(1.5707963267948966) q[2], q[4];
p(0) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
rzz(1.5707963267948966) q[9], q[5];
rz(1.5707963267948966) q[3];
sdg q[0];
id q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
s q[5];
rz(1.5707963267948966) q[8];
p(0) q[1];
rx(1.5707963267948966) q[7];
p(0) q[0];
id q[6];
t q[4];
p(0) q[7];
t q[7];
rz(1.5707963267948966) q[0];
tdg q[8];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
t q[2];
u1(1.5707963267948966) q[8];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[0];
t q[7];
tdg q[4];
id q[8];
s q[4];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[8];
cx q[3], q[0];
rz(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[5];
sdg q[6];
tdg q[0];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[1];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
h q[4];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[5];
p(0) q[5];
id q[5];
id q[1];
ry(1.5707963267948966) q[9];
cy q[8], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[6];
rxx(0) q[9], q[6];
swap q[9], q[2];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[6];
tdg q[3];
u1(1.5707963267948966) q[9];
rx(1.5707963267948966) q[4];
sdg q[2];
sdg q[3];
sdg q[5];
cz q[1], q[0];
sdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[0];
id q[4];
sdg q[0];
p(0) q[7];
ry(1.5707963267948966) q[1];
cz q[4], q[5];
cz q[6], q[9];
rz(1.5707963267948966) q[6];
cy q[6], q[7];
s q[9];
t q[5];
swap q[2], q[3];
u1(1.5707963267948966) q[4];
cy q[5], q[7];
rx(1.5707963267948966) q[9];
t q[8];
sdg q[1];
rzz(1.5707963267948966) q[3], q[0];
s q[2];
sdg q[9];
rxx(0) q[9], q[4];
tdg q[5];
h q[0];
swap q[9], q[3];
u1(1.5707963267948966) q[5];
tdg q[3];
id q[1];
cz q[2], q[7];
rz(1.5707963267948966) q[2];
h q[1];
tdg q[9];
u3(0, 0, 1.5707963267948966) q[8];
h q[1];
sdg q[1];
id q[4];
rxx(0) q[0], q[7];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[4];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[5];
id q[3];
rzz(1.5707963267948966) q[6], q[9];
tdg q[1];
h q[3];
u3(0, 0, 1.5707963267948966) q[8];
cx q[7], q[3];
p(0) q[7];
p(0) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[4];
rz(1.5707963267948966) q[1];
tdg q[0];
sdg q[0];
h q[3];
rz(1.5707963267948966) q[8];
cz q[5], q[3];
u3(0, 0, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
cu1(1.5707963267948966) q[4], q[1];
sdg q[6];
p(0) q[5];