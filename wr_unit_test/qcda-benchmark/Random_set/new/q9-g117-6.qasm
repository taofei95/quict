OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
id q[5];
h q[3];
rx(1.5707963267948966) q[4];
t q[0];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[7];
cx q[1], q[6];
crz(1.5707963267948966) q[1], q[8];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
tdg q[1];
t q[1];
t q[1];
h q[6];
tdg q[5];
rxx(0) q[0], q[6];
u1(1.5707963267948966) q[0];
tdg q[2];
tdg q[3];
sdg q[6];
id q[2];
u3(0, 0, 1.5707963267948966) q[6];
rxx(0) q[6], q[3];
ry(1.5707963267948966) q[0];
cz q[2], q[0];
s q[1];
u3(0, 0, 1.5707963267948966) q[4];
id q[7];
cu1(1.5707963267948966) q[1], q[2];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
ryy(1.5707963267948966) q[3], q[5];
rxx(0) q[0], q[7];
id q[1];
x q[1];
p(0) q[7];
p(0) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
crz(1.5707963267948966) q[6], q[8];
cu3(1.5707963267948966, 0, 0) q[5], q[2];
cu3(1.5707963267948966, 0, 0) q[5], q[6];
t q[7];
cz q[0], q[2];
x q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[8];
id q[4];
t q[7];
crz(1.5707963267948966) q[0], q[3];
ry(1.5707963267948966) q[8];
s q[6];
u3(0, 0, 1.5707963267948966) q[2];
id q[3];
sdg q[0];
h q[7];
u1(1.5707963267948966) q[1];
h q[0];
swap q[2], q[8];
ry(1.5707963267948966) q[7];
id q[6];
p(0) q[2];
u1(1.5707963267948966) q[0];
p(0) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[8];
crz(1.5707963267948966) q[0], q[3];
u3(0, 0, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
sdg q[5];
swap q[0], q[8];
x q[1];
cx q[8], q[2];
p(0) q[1];
cu3(1.5707963267948966, 0, 0) q[8], q[5];
id q[0];
sdg q[7];
x q[5];
u1(1.5707963267948966) q[2];
h q[5];
ch q[6], q[2];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
rxx(0) q[6], q[0];
id q[8];
cy q[6], q[7];
rxx(0) q[8], q[1];
id q[0];
cy q[6], q[3];
x q[7];
rxx(0) q[0], q[8];
u2(1.5707963267948966, 1.5707963267948966) q[4];
h q[1];
cu1(1.5707963267948966) q[3], q[2];
sdg q[2];
h q[1];
p(0) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[6];
u1(1.5707963267948966) q[7];
x q[4];
p(0) q[7];
cy q[4], q[3];
p(0) q[1];
h q[1];
u1(1.5707963267948966) q[6];
cu1(1.5707963267948966) q[1], q[6];
s q[3];
rz(1.5707963267948966) q[2];