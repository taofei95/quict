OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(1.5707963267948966) q[0];
cz q[4], q[9];
swap q[9], q[3];
u3(0, 0, 1.5707963267948966) q[9];
ch q[8], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[4];
ryy(1.5707963267948966) q[3], q[0];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
sdg q[0];
rxx(0) q[8], q[7];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[7];
u1(1.5707963267948966) q[8];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[4];
x q[7];
rz(1.5707963267948966) q[0];
cu1(1.5707963267948966) q[6], q[7];
tdg q[0];
x q[0];
u3(0, 0, 1.5707963267948966) q[3];
h q[8];
cy q[7], q[4];
id q[0];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[7];
sdg q[2];
h q[9];
rx(1.5707963267948966) q[3];
ch q[5], q[1];
p(0) q[3];
p(0) q[0];
id q[2];
t q[6];
u3(0, 0, 1.5707963267948966) q[6];
x q[1];
ch q[4], q[9];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
t q[8];
p(0) q[9];
x q[2];
tdg q[2];
sdg q[2];
s q[2];
h q[2];
cu3(1.5707963267948966, 0, 0) q[3], q[2];
id q[2];
p(0) q[0];
tdg q[8];
cu1(1.5707963267948966) q[3], q[0];
p(0) q[9];
cu3(1.5707963267948966, 0, 0) q[2], q[4];
cx q[5], q[1];
u3(0, 0, 1.5707963267948966) q[1];
x q[1];
u2(1.5707963267948966, 1.5707963267948966) q[9];
id q[9];
id q[0];
rx(1.5707963267948966) q[5];
tdg q[0];
p(0) q[4];
rx(1.5707963267948966) q[8];
rxx(0) q[2], q[8];
x q[4];
p(0) q[6];
rxx(0) q[1], q[9];
rzz(1.5707963267948966) q[2], q[3];
ry(1.5707963267948966) q[7];
x q[7];
cu3(1.5707963267948966, 0, 0) q[8], q[5];
rx(1.5707963267948966) q[3];
tdg q[1];
s q[6];
cy q[7], q[2];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
t q[6];
x q[5];
ry(1.5707963267948966) q[1];
tdg q[0];
cu1(1.5707963267948966) q[8], q[1];
x q[2];
ry(1.5707963267948966) q[8];
ch q[7], q[8];
id q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
tdg q[9];
sdg q[8];
sdg q[4];
ryy(1.5707963267948966) q[9], q[5];
s q[6];
h q[7];
p(0) q[7];
h q[2];
s q[3];
ryy(1.5707963267948966) q[6], q[5];
cu1(1.5707963267948966) q[4], q[8];
crz(1.5707963267948966) q[8], q[3];
sdg q[9];
t q[4];
id q[7];
ry(1.5707963267948966) q[9];
p(0) q[4];
p(0) q[2];