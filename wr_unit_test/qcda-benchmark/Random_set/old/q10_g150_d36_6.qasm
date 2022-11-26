OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
x q[3];
rz(1.5707963267948966) q[0];
s q[2];
h q[3];
u1(1.5707963267948966) q[2];
sdg q[5];
p(0) q[5];
rx(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[9];
cx q[0], q[5];
sdg q[5];
rxx(0) q[3], q[0];
id q[1];
cu1(1.5707963267948966) q[0], q[4];
rz(1.5707963267948966) q[0];
x q[5];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
ch q[3], q[5];
ryy(1.5707963267948966) q[5], q[6];
x q[5];
rx(1.5707963267948966) q[3];
cx q[6], q[7];
cu1(1.5707963267948966) q[8], q[6];
rz(1.5707963267948966) q[3];
id q[5];
u2(1.5707963267948966, 1.5707963267948966) q[2];
tdg q[4];
swap q[3], q[2];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[2];
sdg q[1];
cz q[1], q[4];
tdg q[9];
h q[1];
cu3(1.5707963267948966, 0, 0) q[2], q[8];
x q[5];
rz(1.5707963267948966) q[7];
t q[1];
ryy(1.5707963267948966) q[5], q[6];
p(0) q[0];
p(0) q[3];
crz(1.5707963267948966) q[3], q[0];
h q[5];
swap q[7], q[4];
s q[3];
id q[3];
x q[2];
p(0) q[6];
ch q[4], q[5];
u1(1.5707963267948966) q[7];
swap q[7], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[8];
cx q[9], q[4];
rx(1.5707963267948966) q[5];
id q[5];
cy q[6], q[7];
id q[6];
t q[3];
u3(0, 0, 1.5707963267948966) q[8];
cu1(1.5707963267948966) q[5], q[1];
sdg q[5];
ry(1.5707963267948966) q[4];
cz q[3], q[0];
ry(1.5707963267948966) q[5];
h q[0];
tdg q[1];
rxx(0) q[8], q[5];
u1(1.5707963267948966) q[6];
sdg q[7];
s q[3];
s q[1];
ry(1.5707963267948966) q[8];
t q[0];
u3(0, 0, 1.5707963267948966) q[2];
crz(1.5707963267948966) q[1], q[3];
t q[6];
u1(1.5707963267948966) q[0];
s q[3];
sdg q[7];
u1(1.5707963267948966) q[1];
p(0) q[3];
x q[5];
cx q[9], q[5];
h q[9];
rzz(1.5707963267948966) q[2], q[6];
rz(1.5707963267948966) q[9];
h q[9];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[8];
rzz(1.5707963267948966) q[5], q[3];
u3(0, 0, 1.5707963267948966) q[9];
x q[9];
h q[8];
ryy(1.5707963267948966) q[4], q[1];
t q[2];
ry(1.5707963267948966) q[9];
cz q[6], q[8];
ryy(1.5707963267948966) q[2], q[1];
x q[2];
tdg q[4];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[4];
ryy(1.5707963267948966) q[0], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
x q[4];
cu3(1.5707963267948966, 0, 0) q[0], q[6];
id q[7];
p(0) q[3];
u3(0, 0, 1.5707963267948966) q[1];
x q[9];
t q[2];
h q[3];
u3(0, 0, 1.5707963267948966) q[0];
x q[3];
x q[2];
sdg q[5];
sdg q[3];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[6];
cz q[5], q[2];
tdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[9];
s q[5];
p(0) q[0];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
cu1(1.5707963267948966) q[9], q[6];
s q[8];
crz(1.5707963267948966) q[7], q[2];
u3(0, 0, 1.5707963267948966) q[6];
tdg q[6];
s q[5];
cu3(1.5707963267948966, 0, 0) q[6], q[1];
rz(1.5707963267948966) q[2];
rxx(0) q[7], q[8];
cu1(1.5707963267948966) q[6], q[4];
id q[0];
tdg q[0];
x q[6];
t q[2];
rz(1.5707963267948966) q[2];
sdg q[5];
sdg q[0];
ry(1.5707963267948966) q[4];
cu3(1.5707963267948966, 0, 0) q[8], q[6];
crz(1.5707963267948966) q[2], q[5];