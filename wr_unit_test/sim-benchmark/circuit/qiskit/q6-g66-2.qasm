OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
cy q[0], q[3];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
sdg q[1];
p(0) q[2];
rz(1.5707963267948966) q[3];
t q[2];
p(0) q[5];
u1(1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
s q[0];
cu3(1.5707963267948966, 0, 0) q[4], q[1];
rz(1.5707963267948966) q[1];
h q[0];
p(0) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
rzz(1.5707963267948966) q[0], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
tdg q[1];
p(0) q[4];
u3(0, 0, 1.5707963267948966) q[4];
s q[2];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[0];
h q[1];
id q[1];
h q[5];
s q[4];
h q[1];
h q[1];
sdg q[0];
rz(1.5707963267948966) q[2];
ch q[4], q[1];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
p(0) q[4];
rx(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
h q[1];
u3(0, 0, 1.5707963267948966) q[0];
rxx(0) q[0], q[3];
u3(0, 0, 1.5707963267948966) q[5];
cu3(1.5707963267948966, 0, 0) q[4], q[2];
id q[0];
h q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
s q[5];
cu1(1.5707963267948966) q[2], q[3];
rx(1.5707963267948966) q[2];
cy q[4], q[0];
ch q[4], q[3];
h q[3];
p(0) q[4];
id q[2];
rx(1.5707963267948966) q[5];
sdg q[3];
cy q[2], q[4];
p(0) q[5];
ry(1.5707963267948966) q[0];
cz q[4], q[3];
rx(1.5707963267948966) q[4];
p(0) q[2];