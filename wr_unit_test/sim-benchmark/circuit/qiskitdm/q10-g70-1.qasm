OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
sdg q[0];
u3(0, 0, 1.5707963267948966) q[9];
cx q[8], q[9];
s q[0];
t q[4];
ry(1.5707963267948966) q[8];
p(0) q[7];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
cu1(1.5707963267948966) q[7], q[8];
cx q[3], q[5];
ry(1.5707963267948966) q[3];
tdg q[3];
s q[1];
u3(0, 0, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
t q[6];
cu3(1.5707963267948966, 0, 0) q[6], q[3];
p(0) q[0];
cx q[7], q[5];
h q[6];
h q[0];
id q[2];
u3(0, 0, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
cx q[1], q[4];
t q[1];
id q[8];
rx(1.5707963267948966) q[8];
id q[4];
u1(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[2];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
u1(1.5707963267948966) q[3];
s q[5];
swap q[0], q[9];
p(0) q[9];
rxx(0) q[3], q[9];
t q[2];
id q[8];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
cu3(1.5707963267948966, 0, 0) q[7], q[9];
sdg q[0];
swap q[0], q[8];
id q[7];
ry(1.5707963267948966) q[7];
u1(1.5707963267948966) q[4];
id q[2];
tdg q[1];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[7];
t q[8];
rz(1.5707963267948966) q[7];
h q[5];
rx(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[2];
id q[8];
sdg q[9];
p(0) q[5];
rz(1.5707963267948966) q[0];
sdg q[0];
cu3(1.5707963267948966, 0, 0) q[9], q[7];
rx(1.5707963267948966) q[0];