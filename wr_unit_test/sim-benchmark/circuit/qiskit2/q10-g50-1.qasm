OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
cy q[4], q[8];
rzz(1.5707963267948966) q[4], q[0];
p(0) q[7];
swap q[8], q[4];
rz(1.5707963267948966) q[3];
t q[5];
p(0) q[7];
rx(1.5707963267948966) q[1];
s q[0];
s q[2];
sdg q[4];
cy q[4], q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[2];
tdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rxx(0) q[8], q[2];
ry(1.5707963267948966) q[0];
tdg q[2];
s q[9];
t q[5];
t q[9];
u3(0, 0, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
t q[7];
cz q[0], q[5];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[7];
h q[9];
p(0) q[6];
p(0) q[6];
ry(1.5707963267948966) q[3];
rxx(0) q[1], q[4];
cx q[9], q[4];
h q[7];
h q[7];
swap q[7], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rzz(1.5707963267948966) q[4], q[0];
t q[0];
ry(1.5707963267948966) q[4];
cx q[1], q[7];
rzz(1.5707963267948966) q[1], q[7];
swap q[8], q[1];
u1(1.5707963267948966) q[6];
p(0) q[0];
swap q[9], q[1];
p(0) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];