OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
u3(0, 0, 1.5707963267948966) q[0];
h q[4];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
cu1(1.5707963267948966) q[5], q[2];
cy q[3], q[5];
u1(1.5707963267948966) q[5];
sdg q[2];
u1(1.5707963267948966) q[4];
s q[5];
cu1(1.5707963267948966) q[3], q[5];
t q[1];
rxx(0) q[2], q[0];
s q[4];
u3(0, 0, 1.5707963267948966) q[5];
s q[0];
id q[5];
u3(0, 0, 1.5707963267948966) q[0];
t q[4];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
sdg q[0];
rzz(1.5707963267948966) q[2], q[3];
u3(0, 0, 1.5707963267948966) q[0];
id q[4];
p(0) q[4];
cz q[0], q[3];
ry(1.5707963267948966) q[2];
sdg q[3];
cu1(1.5707963267948966) q[1], q[0];
cz q[2], q[0];
s q[0];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
h q[4];
p(0) q[5];
ry(1.5707963267948966) q[2];
s q[4];
tdg q[0];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
h q[2];
u1(1.5707963267948966) q[2];
cx q[0], q[3];
cy q[4], q[5];
ry(1.5707963267948966) q[5];
id q[5];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
id q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rzz(1.5707963267948966) q[2], q[1];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
swap q[2], q[5];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
h q[2];
p(0) q[4];
id q[3];
s q[2];
id q[3];
tdg q[2];
t q[5];
rx(1.5707963267948966) q[4];
rzz(1.5707963267948966) q[4], q[3];
u1(1.5707963267948966) q[4];
sdg q[0];
rx(1.5707963267948966) q[0];
id q[1];
cx q[0], q[1];
ry(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[5];
swap q[1], q[3];
u3(0, 0, 1.5707963267948966) q[5];
sdg q[5];
sdg q[0];
rzz(1.5707963267948966) q[4], q[0];
p(0) q[0];
id q[1];
h q[5];
p(0) q[5];
sdg q[1];
cu1(1.5707963267948966) q[2], q[4];
rz(1.5707963267948966) q[0];
s q[0];