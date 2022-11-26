OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
p(0) q[2];
ry(1.5707963267948966) q[7];
tdg q[9];
ry(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[8];
p(0) q[2];
id q[9];
ry(1.5707963267948966) q[1];
id q[0];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[6], q[9];
x q[1];
x q[5];
tdg q[6];
tdg q[2];
rx(1.5707963267948966) q[6];
sdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[4];
ryy(1.5707963267948966) q[9], q[1];
s q[2];
id q[0];
ry(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
h q[3];
t q[9];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
sdg q[1];
cu1(1.5707963267948966) q[2], q[5];
s q[4];
rz(1.5707963267948966) q[1];
h q[0];
p(0) q[0];
t q[6];
p(0) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
cx q[0], q[8];
rz(1.5707963267948966) q[2];
cx q[8], q[5];
id q[7];
crz(1.5707963267948966) q[6], q[0];
cx q[6], q[4];
s q[6];
sdg q[1];
rx(1.5707963267948966) q[1];
tdg q[4];
cz q[8], q[6];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[8];
id q[7];
x q[8];
h q[6];
s q[6];
u3(0, 0, 1.5707963267948966) q[8];
cx q[0], q[6];
u1(1.5707963267948966) q[6];
sdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
p(0) q[6];
x q[1];
u1(1.5707963267948966) q[9];
t q[1];
t q[8];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];
p(0) q[1];
rz(1.5707963267948966) q[6];
t q[5];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[5];
tdg q[4];
p(0) q[9];
cy q[5], q[9];
x q[6];
id q[2];
sdg q[6];
tdg q[6];
ry(1.5707963267948966) q[5];
id q[1];
cz q[3], q[8];
ry(1.5707963267948966) q[0];
s q[8];
tdg q[7];
s q[7];
rx(1.5707963267948966) q[2];
tdg q[4];
cy q[4], q[3];
x q[3];
s q[1];
tdg q[2];
rz(1.5707963267948966) q[8];
h q[1];
rxx(0) q[9], q[0];
ryy(1.5707963267948966) q[0], q[1];
crz(1.5707963267948966) q[1], q[0];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[8];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[7];
x q[5];
p(0) q[1];
cu1(1.5707963267948966) q[3], q[7];
sdg q[7];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
h q[6];
sdg q[3];
t q[3];
p(0) q[5];
cy q[0], q[4];
u1(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
ch q[2], q[1];
rx(1.5707963267948966) q[2];
s q[8];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[5];
sdg q[0];
t q[0];
x q[2];
cz q[6], q[8];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
sdg q[5];
t q[8];
id q[2];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[2];
id q[6];
cy q[4], q[1];
u1(1.5707963267948966) q[4];
p(0) q[8];
u3(0, 0, 1.5707963267948966) q[3];
cy q[6], q[9];
u2(1.5707963267948966, 1.5707963267948966) q[1];
x q[9];
p(0) q[3];
h q[1];
crz(1.5707963267948966) q[0], q[6];
sdg q[7];
h q[2];