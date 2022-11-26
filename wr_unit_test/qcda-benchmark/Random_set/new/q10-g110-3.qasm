OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
u1(1.5707963267948966) q[0];
x q[1];
x q[9];
rx(1.5707963267948966) q[7];
tdg q[9];
cz q[5], q[7];
ry(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[4];
rz(1.5707963267948966) q[5];
ch q[1], q[9];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[9];
tdg q[2];
ryy(1.5707963267948966) q[5], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[9];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
s q[2];
rx(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
sdg q[3];
rz(1.5707963267948966) q[6];
id q[9];
id q[2];
x q[6];
x q[3];
tdg q[4];
rxx(0) q[0], q[2];
t q[4];
u3(0, 0, 1.5707963267948966) q[2];
h q[2];
id q[5];
u3(0, 0, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
x q[2];
id q[4];
t q[2];
cx q[5], q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[3];
u1(1.5707963267948966) q[9];
h q[1];
sdg q[7];
ry(1.5707963267948966) q[7];
sdg q[1];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[9];
h q[5];
id q[5];
x q[1];
crz(1.5707963267948966) q[3], q[7];
rx(1.5707963267948966) q[7];
p(0) q[5];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[1];
ryy(1.5707963267948966) q[9], q[7];
cu1(1.5707963267948966) q[4], q[6];
s q[2];
t q[5];
cu1(1.5707963267948966) q[5], q[2];
sdg q[4];
t q[7];
s q[4];
crz(1.5707963267948966) q[8], q[1];
cy q[6], q[4];
rxx(0) q[8], q[1];
ry(1.5707963267948966) q[6];
x q[8];
tdg q[8];
x q[6];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[4];
sdg q[2];
swap q[0], q[6];
rx(1.5707963267948966) q[3];
rxx(0) q[5], q[8];
id q[2];
x q[0];
rxx(0) q[8], q[1];
tdg q[2];
rz(1.5707963267948966) q[6];
u1(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[7];
id q[1];
ry(1.5707963267948966) q[8];
tdg q[1];
cy q[2], q[9];
rx(1.5707963267948966) q[9];
swap q[0], q[2];
rx(1.5707963267948966) q[9];
swap q[3], q[5];
cy q[9], q[8];
x q[5];
t q[7];
u3(0, 0, 1.5707963267948966) q[5];
cy q[6], q[8];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cu3(1.5707963267948966, 0, 0) q[2], q[8];
sdg q[9];
sdg q[2];
t q[0];
cz q[4], q[2];