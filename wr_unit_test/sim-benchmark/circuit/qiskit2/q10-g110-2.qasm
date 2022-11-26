OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
t q[9];
p(0) q[3];
cz q[9], q[4];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[8];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[8], q[0];
id q[4];
cu1(1.5707963267948966) q[1], q[6];
h q[5];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[9];
h q[5];
cu1(1.5707963267948966) q[9], q[4];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[0];
h q[8];
t q[2];
s q[4];
sdg q[3];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rzz(1.5707963267948966) q[6], q[5];
u3(0, 0, 1.5707963267948966) q[1];
t q[3];
sdg q[4];
t q[8];
rx(1.5707963267948966) q[9];
id q[5];
rzz(1.5707963267948966) q[4], q[8];
cy q[6], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[9];
cy q[9], q[0];
sdg q[4];
tdg q[6];
p(0) q[7];
cu1(1.5707963267948966) q[9], q[0];
tdg q[1];
cx q[0], q[5];
p(0) q[2];
t q[0];
cu1(1.5707963267948966) q[1], q[8];
rz(1.5707963267948966) q[9];
u1(1.5707963267948966) q[2];
s q[9];
rz(1.5707963267948966) q[4];
rzz(1.5707963267948966) q[7], q[2];
p(0) q[0];
t q[5];
rx(1.5707963267948966) q[4];
tdg q[1];
rx(1.5707963267948966) q[8];
cu1(1.5707963267948966) q[9], q[2];
cy q[9], q[3];
sdg q[4];
id q[8];
rzz(1.5707963267948966) q[9], q[0];
tdg q[1];
cu1(1.5707963267948966) q[7], q[3];
tdg q[9];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[5];
s q[7];
u1(1.5707963267948966) q[0];
h q[4];
rx(1.5707963267948966) q[3];
cz q[2], q[7];
rxx(0) q[3], q[7];
t q[3];
p(0) q[2];
h q[9];
h q[8];
u3(0, 0, 1.5707963267948966) q[8];
id q[8];
u3(0, 0, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[2];
tdg q[3];
cx q[7], q[2];
id q[1];
id q[0];
t q[8];
sdg q[4];
cx q[8], q[0];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[1];
id q[7];
p(0) q[4];
tdg q[2];
rx(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[0], q[9];
sdg q[1];
rx(1.5707963267948966) q[3];
id q[1];
rz(1.5707963267948966) q[9];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];