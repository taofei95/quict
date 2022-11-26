OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
h q[7];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[4];
cu3(1.5707963267948966, 0, 0) q[2], q[6];
cx q[5], q[7];
t q[4];
tdg q[4];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[7];
cu1(1.5707963267948966) q[4], q[7];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
swap q[6], q[1];
ch q[6], q[0];
u3(0, 0, 1.5707963267948966) q[7];
id q[3];
rx(1.5707963267948966) q[5];
rzz(1.5707963267948966) q[5], q[1];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
t q[7];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[7];
u1(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[0];
t q[0];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[1];
id q[1];
rx(1.5707963267948966) q[7];
crz(1.5707963267948966) q[7], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[7];
cu1(1.5707963267948966) q[3], q[5];
sdg q[8];
cz q[1], q[2];
h q[3];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[3];
t q[1];
tdg q[2];
s q[5];
tdg q[2];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[7];
u3(0, 0, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
tdg q[7];
h q[5];
p(0) q[1];
h q[1];
sdg q[3];
h q[6];
h q[1];
sdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[1];
tdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[1];
p(0) q[2];
tdg q[0];
tdg q[3];
ch q[1], q[7];
sdg q[3];
p(0) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
id q[7];
t q[7];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[5];
t q[7];