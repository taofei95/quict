OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
h q[7];
h q[1];
t q[8];
p(0) q[6];
ry(1.5707963267948966) q[8];
id q[4];
s q[8];
s q[9];
x q[5];
id q[7];
ch q[1], q[5];
rz(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[6];
t q[3];
u3(0, 0, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
cu3(1.5707963267948966, 0, 0) q[3], q[2];
u1(1.5707963267948966) q[7];
x q[5];
id q[0];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[8];
sdg q[2];
ry(1.5707963267948966) q[0];
h q[4];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ch q[2], q[1];
sdg q[7];
u1(1.5707963267948966) q[8];
t q[6];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[7];
x q[0];
cy q[1], q[8];
u1(1.5707963267948966) q[4];
id q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[7];
id q[8];
ryy(1.5707963267948966) q[8], q[1];
id q[9];
rz(1.5707963267948966) q[4];
cu1(1.5707963267948966) q[5], q[9];
rz(1.5707963267948966) q[0];
p(0) q[0];
p(0) q[8];
cx q[5], q[4];
x q[8];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[2];
cz q[7], q[2];
rx(1.5707963267948966) q[5];
t q[7];
crz(1.5707963267948966) q[7], q[8];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
h q[7];
cz q[9], q[0];
rz(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[8], q[2];
p(0) q[5];
u3(0, 0, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
h q[1];
tdg q[9];
rx(1.5707963267948966) q[9];
t q[7];
rz(1.5707963267948966) q[2];
tdg q[7];
cz q[6], q[3];
s q[9];
t q[1];
cx q[3], q[9];
t q[3];
crz(1.5707963267948966) q[5], q[7];
rz(1.5707963267948966) q[7];
t q[8];
id q[3];
u3(0, 0, 1.5707963267948966) q[7];
t q[3];
cu3(1.5707963267948966, 0, 0) q[0], q[4];
s q[4];