OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
ch q[7], q[2];
rz(1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[4];
h q[7];
p(0) q[8];
tdg q[6];
u3(0, 0, 1.5707963267948966) q[4];
cz q[2], q[8];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
swap q[0], q[2];
sdg q[8];
cu1(1.5707963267948966) q[2], q[3];
p(0) q[5];
p(0) q[1];
tdg q[6];
crz(1.5707963267948966) q[5], q[4];
s q[7];
ry(1.5707963267948966) q[8];
ch q[6], q[0];
ry(1.5707963267948966) q[2];
sdg q[6];
id q[4];
p(0) q[0];
sdg q[1];
cz q[3], q[1];
rx(1.5707963267948966) q[3];
sdg q[4];
t q[7];
h q[6];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
s q[1];
cy q[3], q[5];
rz(1.5707963267948966) q[3];
cy q[2], q[4];
sdg q[1];
s q[1];
ry(1.5707963267948966) q[0];
sdg q[8];
u1(1.5707963267948966) q[8];
tdg q[6];
cy q[3], q[2];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rx(1.5707963267948966) q[8];
sdg q[4];
p(0) q[0];
cu3(1.5707963267948966, 0, 0) q[4], q[3];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[4];
tdg q[6];
sdg q[6];
u1(1.5707963267948966) q[8];
h q[1];
tdg q[6];
id q[0];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
p(0) q[1];
rx(1.5707963267948966) q[6];
id q[6];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
h q[5];
crz(1.5707963267948966) q[7], q[3];
tdg q[5];
crz(1.5707963267948966) q[5], q[1];
ry(1.5707963267948966) q[6];
id q[3];
ch q[7], q[3];
cy q[0], q[2];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[1];
id q[3];
t q[8];
t q[0];
p(0) q[7];
t q[5];
rx(1.5707963267948966) q[5];
cx q[7], q[2];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[8];
id q[8];
u3(0, 0, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[7];
t q[7];
crz(1.5707963267948966) q[8], q[2];
crz(1.5707963267948966) q[1], q[5];
cx q[0], q[6];
tdg q[4];
t q[0];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[6];
h q[5];
rx(1.5707963267948966) q[8];
tdg q[6];