OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
s q[6];
s q[8];
rx(1.5707963267948966) q[6];
cu3(1.5707963267948966, 0, 0) q[3], q[6];
p(0) q[6];
id q[8];
rz(1.5707963267948966) q[4];
cx q[7], q[8];
cu1(1.5707963267948966) q[9], q[1];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[4];
h q[7];
h q[8];
u3(0, 0, 1.5707963267948966) q[8];
h q[8];
u3(0, 0, 1.5707963267948966) q[4];
p(0) q[3];
tdg q[0];
p(0) q[0];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[5];
id q[1];
u3(0, 0, 1.5707963267948966) q[7];
t q[0];
crz(1.5707963267948966) q[6], q[9];
u1(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[4];
cx q[6], q[8];
rz(1.5707963267948966) q[3];
p(0) q[1];
cx q[0], q[2];
p(0) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rxx(0) q[2], q[4];
tdg q[4];
t q[3];
cy q[6], q[5];
rx(1.5707963267948966) q[0];
h q[2];
tdg q[9];
h q[3];
u1(1.5707963267948966) q[4];
p(0) q[2];
id q[8];
tdg q[9];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[9];
sdg q[0];
s q[6];
t q[7];
rz(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[1];
s q[0];
tdg q[6];
rxx(0) q[9], q[8];
rx(1.5707963267948966) q[3];
p(0) q[3];
s q[6];
rz(1.5707963267948966) q[4];
swap q[6], q[3];
s q[1];
rz(1.5707963267948966) q[0];
h q[4];
ch q[3], q[7];
tdg q[7];
rxx(0) q[6], q[2];
u1(1.5707963267948966) q[1];
tdg q[8];
h q[4];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[7];
rx(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[6];
s q[7];
u2(1.5707963267948966, 1.5707963267948966) q[9];
tdg q[6];
s q[0];
h q[9];
cu1(1.5707963267948966) q[9], q[6];
p(0) q[4];
p(0) q[2];
p(0) q[2];
s q[3];
s q[1];
p(0) q[6];
crz(1.5707963267948966) q[7], q[2];
p(0) q[3];
p(0) q[7];
rx(1.5707963267948966) q[5];
id q[4];
cx q[0], q[6];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
p(0) q[5];
sdg q[0];
id q[8];
s q[4];
rx(1.5707963267948966) q[4];
h q[1];
ry(1.5707963267948966) q[8];
tdg q[2];
rz(1.5707963267948966) q[0];
cx q[7], q[6];
ry(1.5707963267948966) q[0];
h q[1];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[7];
sdg q[6];
p(0) q[5];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[8];
cx q[1], q[3];
t q[2];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[5];
t q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
crz(1.5707963267948966) q[5], q[1];
t q[4];