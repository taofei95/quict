OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
p(0) q[9];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[2];
sdg q[3];
h q[6];
h q[2];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
h q[11];
id q[8];
tdg q[6];
cu1(1.5707963267948966) q[1], q[8];
ry(1.5707963267948966) q[10];
id q[7];
rx(1.5707963267948966) q[2];
t q[2];
s q[3];
cu1(1.5707963267948966) q[10], q[8];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[11];
t q[6];
rz(1.5707963267948966) q[9];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[3];
rxx(0) q[2], q[8];
cu3(1.5707963267948966, 0, 0) q[10], q[4];
t q[11];
tdg q[3];
h q[7];
u2(1.5707963267948966, 1.5707963267948966) q[6];
h q[4];
cz q[8], q[6];
p(0) q[9];
cu1(1.5707963267948966) q[5], q[6];
p(0) q[10];
sdg q[6];
ry(1.5707963267948966) q[3];
h q[9];
u1(1.5707963267948966) q[8];
id q[0];
s q[2];
t q[9];
cz q[8], q[1];
h q[0];
ry(1.5707963267948966) q[4];
cy q[5], q[2];
t q[3];
ry(1.5707963267948966) q[2];
t q[3];
ry(1.5707963267948966) q[6];
cu3(1.5707963267948966, 0, 0) q[8], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
id q[7];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
swap q[11], q[0];
cu1(1.5707963267948966) q[7], q[5];
rzz(1.5707963267948966) q[2], q[3];
u1(1.5707963267948966) q[2];
sdg q[3];
p(0) q[9];
cy q[8], q[3];
id q[8];
rxx(0) q[1], q[6];
sdg q[1];
p(0) q[6];
rxx(0) q[0], q[6];
p(0) q[9];
swap q[10], q[9];
rx(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
tdg q[7];
u1(1.5707963267948966) q[5];
id q[11];
s q[8];
id q[3];
ry(1.5707963267948966) q[10];
rxx(0) q[8], q[2];
id q[7];
sdg q[3];
u3(0, 0, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[2];
sdg q[3];
cy q[5], q[11];
tdg q[6];
cu3(1.5707963267948966, 0, 0) q[5], q[4];
rx(1.5707963267948966) q[5];
s q[11];
u1(1.5707963267948966) q[4];
swap q[0], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
sdg q[8];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[8];
cz q[10], q[7];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[9];
s q[1];
s q[6];
rx(1.5707963267948966) q[8];
tdg q[2];
cu3(1.5707963267948966, 0, 0) q[8], q[4];
tdg q[7];
cu3(1.5707963267948966, 0, 0) q[6], q[5];
rz(1.5707963267948966) q[8];
swap q[0], q[10];
u2(1.5707963267948966, 1.5707963267948966) q[4];
s q[3];
tdg q[8];
id q[8];
cz q[6], q[5];
rxx(0) q[4], q[6];
t q[9];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[1];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[8];
rzz(1.5707963267948966) q[8], q[11];
cu3(1.5707963267948966, 0, 0) q[4], q[11];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[5];
rxx(0) q[5], q[6];
tdg q[8];
id q[0];
sdg q[2];
h q[11];