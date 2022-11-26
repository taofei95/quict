OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
cu3(1.5707963267948966, 0, 0) q[2], q[4];
p(0) q[3];
s q[5];
tdg q[2];
cy q[3], q[4];
rxx(0) q[4], q[2];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
rxx(0) q[0], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
t q[4];
s q[2];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
h q[5];
s q[3];
h q[4];
id q[3];
rz(1.5707963267948966) q[2];
tdg q[4];
id q[0];
s q[5];
p(0) q[2];
p(0) q[2];
p(0) q[4];
s q[3];
cy q[2], q[0];
id q[2];
s q[1];
cu1(1.5707963267948966) q[1], q[4];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
s q[3];
cz q[1], q[2];
cx q[2], q[4];
u1(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
cy q[0], q[3];
cx q[4], q[1];
s q[2];
rz(1.5707963267948966) q[3];
cx q[1], q[4];
p(0) q[2];
rzz(1.5707963267948966) q[5], q[4];
p(0) q[5];
tdg q[0];
rx(1.5707963267948966) q[5];
sdg q[5];
cu1(1.5707963267948966) q[3], q[4];
cu1(1.5707963267948966) q[0], q[2];
tdg q[3];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
sdg q[5];
u1(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
h q[0];
p(0) q[2];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
h q[5];
t q[2];
ry(1.5707963267948966) q[3];
id q[2];
cu1(1.5707963267948966) q[2], q[0];
swap q[5], q[1];
p(0) q[5];
cy q[5], q[0];
tdg q[2];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[2];
rzz(1.5707963267948966) q[0], q[5];
cu3(1.5707963267948966, 0, 0) q[4], q[2];
cz q[5], q[1];
h q[3];
cz q[1], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];