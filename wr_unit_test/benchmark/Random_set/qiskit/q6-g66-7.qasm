OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
id q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rzz(1.5707963267948966) q[5], q[3];
rz(1.5707963267948966) q[5];
cy q[0], q[3];
h q[0];
swap q[0], q[3];
p(0) q[4];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[2];
h q[4];
sdg q[1];
tdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cu3(1.5707963267948966, 0, 0) q[3], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
s q[3];
p(0) q[3];
sdg q[0];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
crz(1.5707963267948966) q[2], q[5];
ry(1.5707963267948966) q[5];
id q[1];
rzz(1.5707963267948966) q[4], q[0];
rzz(1.5707963267948966) q[4], q[5];
s q[1];
t q[4];
u1(1.5707963267948966) q[1];
id q[5];
rx(1.5707963267948966) q[3];
crz(1.5707963267948966) q[2], q[1];
crz(1.5707963267948966) q[1], q[0];
cy q[0], q[4];
h q[3];
id q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
s q[0];
rx(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[1], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[2];
p(0) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
t q[1];
u1(1.5707963267948966) q[4];
swap q[0], q[1];
swap q[5], q[3];
sdg q[3];
tdg q[4];
h q[0];
p(0) q[1];
s q[0];
p(0) q[0];
t q[3];
tdg q[1];
swap q[0], q[4];
u3(0, 0, 1.5707963267948966) q[2];
cy q[4], q[1];
ch q[5], q[0];
s q[3];
rz(1.5707963267948966) q[3];
rzz(1.5707963267948966) q[5], q[4];
p(0) q[1];