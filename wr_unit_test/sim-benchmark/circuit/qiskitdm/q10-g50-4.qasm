OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rzz(1.5707963267948966) q[8], q[4];
t q[0];
rz(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[8];
id q[5];
tdg q[4];
u1(1.5707963267948966) q[8];
id q[7];
rzz(1.5707963267948966) q[1], q[7];
tdg q[4];
t q[0];
tdg q[4];
ry(1.5707963267948966) q[6];
cu3(1.5707963267948966, 0, 0) q[2], q[7];
tdg q[9];
id q[1];
u1(1.5707963267948966) q[4];
tdg q[9];
ry(1.5707963267948966) q[0];
tdg q[6];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[7];
sdg q[6];
sdg q[3];
u1(1.5707963267948966) q[3];
s q[3];
t q[7];
swap q[7], q[5];
s q[6];
u3(0, 0, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[4];
id q[4];
sdg q[5];
cu1(1.5707963267948966) q[5], q[8];
t q[2];
p(0) q[4];
rx(1.5707963267948966) q[3];
swap q[4], q[6];
h q[7];
t q[5];
cy q[5], q[2];
rx(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[4];
cx q[3], q[1];
cz q[9], q[3];
rx(1.5707963267948966) q[9];
p(0) q[9];
id q[1];
cu3(1.5707963267948966, 0, 0) q[5], q[7];