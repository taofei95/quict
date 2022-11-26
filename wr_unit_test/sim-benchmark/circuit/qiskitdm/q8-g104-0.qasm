OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
swap q[4], q[0];
rx(1.5707963267948966) q[1];
p(0) q[4];
id q[6];
rx(1.5707963267948966) q[3];
cx q[0], q[5];
h q[7];
u3(0, 0, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[5];
id q[4];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[5];
t q[1];
cz q[2], q[5];
t q[4];
cz q[3], q[6];
cu3(1.5707963267948966, 0, 0) q[2], q[4];
cu3(1.5707963267948966, 0, 0) q[2], q[3];
sdg q[3];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[4];
h q[6];
ry(1.5707963267948966) q[2];
h q[4];
tdg q[1];
rzz(1.5707963267948966) q[4], q[3];
u3(0, 0, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[7];
p(0) q[3];
rx(1.5707963267948966) q[7];
tdg q[4];
rzz(1.5707963267948966) q[5], q[2];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[6], q[3];
rx(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
p(0) q[7];
s q[1];
cz q[5], q[7];
h q[1];
rz(1.5707963267948966) q[3];
id q[3];
s q[1];
u1(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
id q[7];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[1];
id q[5];
t q[3];
ry(1.5707963267948966) q[1];
sdg q[2];
cu3(1.5707963267948966, 0, 0) q[3], q[2];
u3(0, 0, 1.5707963267948966) q[2];
t q[2];
rz(1.5707963267948966) q[1];
s q[2];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[1];
h q[4];
tdg q[3];
tdg q[2];
tdg q[3];
tdg q[4];
t q[5];
ry(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[7];
rz(1.5707963267948966) q[1];
swap q[0], q[7];
s q[1];
tdg q[2];
cz q[7], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[2];
s q[0];
ry(1.5707963267948966) q[6];
sdg q[3];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rzz(1.5707963267948966) q[5], q[4];
ry(1.5707963267948966) q[5];
id q[3];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[3];
t q[3];
u3(0, 0, 1.5707963267948966) q[5];
p(0) q[7];
h q[7];
rx(1.5707963267948966) q[3];
id q[3];
tdg q[1];
h q[5];
rx(1.5707963267948966) q[3];
cy q[7], q[4];
rx(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[0];
p(0) q[7];