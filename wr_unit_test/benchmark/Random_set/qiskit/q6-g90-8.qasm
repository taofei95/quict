OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
t q[5];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
cy q[0], q[5];
s q[1];
s q[3];
u1(1.5707963267948966) q[1];
ch q[0], q[4];
ry(1.5707963267948966) q[2];
p(0) q[3];
h q[0];
sdg q[1];
sdg q[0];
h q[3];
u1(1.5707963267948966) q[5];
ry(1.5707963267948966) q[1];
p(0) q[2];
sdg q[0];
h q[3];
h q[5];
tdg q[4];
cx q[3], q[5];
t q[1];
h q[2];
cz q[2], q[0];
rz(1.5707963267948966) q[2];
p(0) q[4];
u3(0, 0, 1.5707963267948966) q[1];
rzz(1.5707963267948966) q[4], q[1];
swap q[3], q[4];
p(0) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[1];
ch q[2], q[4];
crz(1.5707963267948966) q[1], q[3];
rx(1.5707963267948966) q[0];
t q[3];
p(0) q[4];
u1(1.5707963267948966) q[3];
t q[5];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[3];
t q[5];
u3(0, 0, 1.5707963267948966) q[3];
h q[2];
rz(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[2];
id q[4];
sdg q[2];
rz(1.5707963267948966) q[2];
t q[1];
id q[4];
sdg q[1];
cz q[1], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
swap q[0], q[2];
t q[2];
u3(0, 0, 1.5707963267948966) q[1];
p(0) q[2];
sdg q[3];
sdg q[1];
crz(1.5707963267948966) q[2], q[1];
tdg q[0];
u3(0, 0, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
tdg q[4];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[5];
s q[4];
cy q[0], q[1];
t q[1];
h q[1];
u1(1.5707963267948966) q[2];
t q[1];
tdg q[4];
rx(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[4];
sdg q[4];
sdg q[1];
swap q[4], q[2];
cz q[0], q[4];
sdg q[3];
rxx(0) q[1], q[3];