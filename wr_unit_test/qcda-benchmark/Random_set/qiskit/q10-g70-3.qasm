OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[5];
p(0) q[7];
u1(1.5707963267948966) q[9];
cx q[7], q[0];
t q[8];
s q[4];
cu3(1.5707963267948966, 0, 0) q[7], q[8];
cx q[0], q[5];
ry(1.5707963267948966) q[6];
rxx(0) q[1], q[2];
sdg q[4];
p(0) q[8];
swap q[1], q[7];
p(0) q[9];
cx q[8], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
tdg q[0];
u1(1.5707963267948966) q[8];
sdg q[1];
t q[6];
rx(1.5707963267948966) q[4];
rzz(1.5707963267948966) q[5], q[3];
rx(1.5707963267948966) q[8];
tdg q[9];
sdg q[0];
ch q[6], q[1];
ch q[1], q[4];
h q[4];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[5];
cx q[4], q[7];
h q[6];
ry(1.5707963267948966) q[5];
s q[9];
cu1(1.5707963267948966) q[8], q[3];
id q[2];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[8];
rxx(0) q[8], q[0];
id q[9];
tdg q[8];
sdg q[3];
rz(1.5707963267948966) q[2];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
cz q[9], q[3];
id q[5];
s q[4];
s q[0];
ch q[3], q[4];
rx(1.5707963267948966) q[0];
h q[8];
swap q[9], q[2];
rz(1.5707963267948966) q[0];
tdg q[0];
u3(0, 0, 1.5707963267948966) q[4];
t q[9];
p(0) q[6];
crz(1.5707963267948966) q[1], q[2];
rx(1.5707963267948966) q[7];
id q[9];
t q[6];