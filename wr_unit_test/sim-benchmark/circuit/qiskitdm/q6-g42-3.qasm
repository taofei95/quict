OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
rzz(1.5707963267948966) q[5], q[4];
u3(0, 0, 1.5707963267948966) q[1];
sdg q[3];
sdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rzz(1.5707963267948966) q[2], q[1];
id q[5];
rx(1.5707963267948966) q[0];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[0];
s q[5];
cu3(1.5707963267948966, 0, 0) q[2], q[4];
tdg q[2];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[5];
cu3(1.5707963267948966, 0, 0) q[0], q[4];
rz(1.5707963267948966) q[1];
h q[3];
u1(1.5707963267948966) q[2];
p(0) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
t q[1];
cz q[3], q[0];
ry(1.5707963267948966) q[4];
cz q[0], q[3];
cx q[1], q[4];
ry(1.5707963267948966) q[2];
h q[0];
t q[4];
tdg q[2];
cu1(1.5707963267948966) q[4], q[1];
s q[0];
id q[4];
s q[4];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
t q[5];
id q[4];
swap q[2], q[0];
u3(0, 0, 1.5707963267948966) q[2];
id q[1];