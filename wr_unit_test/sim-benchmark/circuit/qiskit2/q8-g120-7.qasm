OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[6];
rx(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
h q[2];
rz(1.5707963267948966) q[3];
id q[4];
u3(0, 0, 1.5707963267948966) q[5];
rxx(0) q[0], q[1];
swap q[4], q[1];
tdg q[4];
swap q[5], q[3];
ry(1.5707963267948966) q[2];
t q[1];
p(0) q[4];
rz(1.5707963267948966) q[7];
s q[1];
sdg q[4];
rz(1.5707963267948966) q[5];
cy q[0], q[7];
u2(1.5707963267948966, 1.5707963267948966) q[5];
h q[6];
tdg q[6];
u3(0, 0, 1.5707963267948966) q[3];
id q[7];
u2(1.5707963267948966, 1.5707963267948966) q[4];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[7];
tdg q[1];
u1(1.5707963267948966) q[5];
rzz(1.5707963267948966) q[2], q[1];
tdg q[3];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[3];
cy q[0], q[1];
rz(1.5707963267948966) q[6];
s q[4];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
p(0) q[6];
u1(1.5707963267948966) q[1];
ry(1.5707963267948966) q[7];
tdg q[0];
cz q[1], q[5];
swap q[6], q[4];
u3(0, 0, 1.5707963267948966) q[1];
h q[0];
rx(1.5707963267948966) q[1];
p(0) q[6];
rzz(1.5707963267948966) q[7], q[4];
swap q[7], q[3];
cx q[2], q[7];
rzz(1.5707963267948966) q[0], q[3];
id q[2];
t q[4];
t q[2];
ry(1.5707963267948966) q[3];
cy q[4], q[0];
u1(1.5707963267948966) q[4];
h q[4];
t q[6];
rx(1.5707963267948966) q[2];
s q[6];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[2];
rxx(0) q[3], q[5];
t q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cx q[2], q[1];
rx(1.5707963267948966) q[7];
sdg q[5];
rz(1.5707963267948966) q[5];
cz q[1], q[7];
swap q[5], q[7];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
cx q[7], q[1];
cz q[1], q[6];
p(0) q[6];
h q[7];
id q[3];
ry(1.5707963267948966) q[6];
swap q[2], q[0];
u3(0, 0, 1.5707963267948966) q[7];
rxx(0) q[0], q[7];
u1(1.5707963267948966) q[6];
sdg q[4];
rz(1.5707963267948966) q[4];
h q[1];
sdg q[5];
cu1(1.5707963267948966) q[7], q[0];
h q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
t q[7];
cx q[6], q[3];
cx q[5], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
tdg q[0];
sdg q[1];
t q[2];
cy q[6], q[4];
rxx(0) q[1], q[4];
sdg q[6];
tdg q[5];
s q[4];
rzz(1.5707963267948966) q[0], q[5];
cx q[5], q[1];
s q[0];
u3(0, 0, 1.5707963267948966) q[4];
s q[6];
swap q[4], q[0];
rxx(0) q[4], q[3];