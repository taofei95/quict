OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
h q[1];
u1(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[3];
rx(1.5707963267948966) q[9];
h q[7];
u3(0, 0, 1.5707963267948966) q[8];
s q[3];
p(0) q[8];
sdg q[9];
sdg q[6];
cy q[8], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rxx(0) q[4], q[7];
rx(1.5707963267948966) q[4];
sdg q[3];
u3(0, 0, 1.5707963267948966) q[3];
p(0) q[5];
id q[4];
u1(1.5707963267948966) q[6];
rxx(0) q[5], q[3];
t q[0];
u3(0, 0, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[2];
p(0) q[2];
swap q[0], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
sdg q[2];
t q[8];
ry(1.5707963267948966) q[2];
swap q[3], q[2];
cz q[2], q[8];
s q[7];
h q[9];
p(0) q[8];
h q[4];
ry(1.5707963267948966) q[4];
id q[0];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[6];
t q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[4];
t q[5];
u3(0, 0, 1.5707963267948966) q[8];
cu3(1.5707963267948966, 0, 0) q[6], q[0];
rx(1.5707963267948966) q[1];
sdg q[5];
u1(1.5707963267948966) q[0];
p(0) q[3];
u3(0, 0, 1.5707963267948966) q[5];
s q[7];
cu1(1.5707963267948966) q[3], q[6];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[9];
s q[2];
u3(0, 0, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[0];
id q[5];
u1(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[3];
swap q[1], q[7];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[7];
h q[2];
h q[9];
t q[2];
ry(1.5707963267948966) q[3];
cu1(1.5707963267948966) q[8], q[0];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[8];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[5], q[0];
rx(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[1];
id q[5];
sdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
s q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[5];
rxx(0) q[2], q[8];
rx(1.5707963267948966) q[0];
sdg q[3];
u1(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[4];
cu3(1.5707963267948966, 0, 0) q[4], q[1];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[5];
sdg q[5];
cz q[8], q[3];
ry(1.5707963267948966) q[8];
cz q[4], q[5];
s q[4];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[8];
p(0) q[2];
t q[7];
id q[0];
rx(1.5707963267948966) q[7];
cz q[3], q[0];
id q[6];
cx q[4], q[6];
rxx(0) q[0], q[7];
rz(1.5707963267948966) q[8];
s q[8];
rx(1.5707963267948966) q[1];
tdg q[1];
u1(1.5707963267948966) q[6];
t q[7];
rx(1.5707963267948966) q[4];
tdg q[9];
rz(1.5707963267948966) q[4];
cu1(1.5707963267948966) q[8], q[7];
p(0) q[7];
p(0) q[9];
rxx(0) q[0], q[9];
rxx(0) q[8], q[0];
u3(0, 0, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
swap q[1], q[7];