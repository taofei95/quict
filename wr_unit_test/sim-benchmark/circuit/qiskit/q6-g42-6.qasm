OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
p(0) q[5];
sdg q[0];
cz q[0], q[3];
cz q[4], q[3];
id q[4];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[5];
cz q[1], q[0];
u1(1.5707963267948966) q[0];
h q[0];
tdg q[3];
sdg q[1];
cz q[2], q[4];
u1(1.5707963267948966) q[1];
rzz(1.5707963267948966) q[2], q[5];
s q[4];
swap q[2], q[1];
h q[5];
p(0) q[4];
rx(1.5707963267948966) q[0];
swap q[3], q[5];
ry(1.5707963267948966) q[3];
h q[3];
rzz(1.5707963267948966) q[3], q[1];
t q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
tdg q[4];
t q[3];
id q[0];
p(0) q[0];
rxx(0) q[1], q[4];
u1(1.5707963267948966) q[1];
ch q[4], q[5];
tdg q[0];
p(0) q[1];
id q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
ry(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
id q[2];
rx(1.5707963267948966) q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[1];