OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
u1(1.5707963267948966) q[6];
rx(1.5707963267948966) q[3];
tdg q[4];
swap q[4], q[0];
tdg q[8];
sdg q[2];
rx(1.5707963267948966) q[0];
rxx(0) q[5], q[2];
crz(1.5707963267948966) q[0], q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
t q[1];
ch q[1], q[3];
ch q[4], q[0];
cy q[5], q[6];
u3(0, 0, 1.5707963267948966) q[6];
p(0) q[7];
s q[8];
rx(1.5707963267948966) q[5];
id q[0];
u3(0, 0, 1.5707963267948966) q[2];
t q[3];
ry(1.5707963267948966) q[3];
cx q[7], q[4];
tdg q[6];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
sdg q[2];
tdg q[3];
ch q[3], q[4];
s q[7];
tdg q[1];
s q[6];
id q[2];
t q[1];
rz(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[6];
u1(1.5707963267948966) q[3];
s q[6];
rx(1.5707963267948966) q[7];
tdg q[8];
cx q[6], q[5];
crz(1.5707963267948966) q[7], q[0];
sdg q[3];
ch q[7], q[1];
cx q[4], q[0];
ry(1.5707963267948966) q[5];
p(0) q[1];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[5];
ch q[3], q[1];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[3];
sdg q[3];
sdg q[0];
sdg q[0];
s q[0];
tdg q[6];
rx(1.5707963267948966) q[5];
t q[8];
sdg q[0];
rz(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
cz q[8], q[4];
p(0) q[8];
id q[5];
cu3(1.5707963267948966, 0, 0) q[5], q[4];
rx(1.5707963267948966) q[3];
t q[4];
u1(1.5707963267948966) q[3];
sdg q[0];
tdg q[3];
p(0) q[8];
p(0) q[2];
ch q[3], q[2];
cz q[8], q[7];
cx q[3], q[8];
cz q[4], q[7];