OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
id q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
cz q[6], q[8];
p(0) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[9];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[3];
sdg q[4];
rz(1.5707963267948966) q[3];
tdg q[2];
rx(1.5707963267948966) q[5];
t q[7];
u2(1.5707963267948966, 1.5707963267948966) q[9];
s q[6];
t q[5];
ry(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[9];
ry(1.5707963267948966) q[7];
t q[6];
p(0) q[0];
u3(0, 0, 1.5707963267948966) q[7];
cy q[5], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[4];
sdg q[6];
cy q[6], q[5];
rx(1.5707963267948966) q[1];
p(0) q[2];
s q[2];
cx q[2], q[7];
u1(1.5707963267948966) q[0];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cy q[9], q[0];
tdg q[6];
rxx(0) q[7], q[9];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[9];
h q[3];
u1(1.5707963267948966) q[1];
t q[6];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
rxx(0) q[5], q[3];
rzz(1.5707963267948966) q[4], q[2];
tdg q[0];
cu3(1.5707963267948966, 0, 0) q[4], q[8];
s q[6];
h q[1];