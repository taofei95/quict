OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
s q[3];
cu1(1.5707963267948966) q[1], q[5];
t q[5];
u3(0, 0, 1.5707963267948966) q[7];
h q[0];
rxx(0) q[6], q[7];
cx q[5], q[7];
h q[3];
cy q[1], q[4];
sdg q[7];
id q[5];
sdg q[5];
u1(1.5707963267948966) q[6];
rxx(0) q[4], q[0];
h q[2];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
id q[5];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[3];
h q[2];
h q[5];
t q[0];
rx(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[3];
sdg q[0];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[5];
p(0) q[1];
p(0) q[4];
u3(0, 0, 1.5707963267948966) q[7];
id q[4];
u3(0, 0, 1.5707963267948966) q[0];
id q[2];
rxx(0) q[2], q[6];
ry(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cz q[6], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[4];
tdg q[6];
u1(1.5707963267948966) q[0];
cx q[0], q[5];
tdg q[0];
p(0) q[5];
id q[4];
sdg q[3];
cy q[6], q[2];
p(0) q[1];
u1(1.5707963267948966) q[1];
swap q[3], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rzz(1.5707963267948966) q[3], q[7];
p(0) q[5];
swap q[4], q[5];
swap q[1], q[7];
u2(1.5707963267948966, 1.5707963267948966) q[4];
id q[2];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
rxx(0) q[4], q[6];
id q[1];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
sdg q[5];
u1(1.5707963267948966) q[6];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[7];
rxx(0) q[5], q[1];
rz(1.5707963267948966) q[3];
h q[3];
sdg q[3];
swap q[7], q[5];
h q[2];
cz q[5], q[7];
rx(1.5707963267948966) q[5];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[6];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[2];
rxx(0) q[6], q[0];
rz(1.5707963267948966) q[3];
id q[1];
p(0) q[1];
rz(1.5707963267948966) q[6];
rzz(1.5707963267948966) q[6], q[0];