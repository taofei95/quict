OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
tdg q[4];
h q[3];
s q[0];
sdg q[6];
tdg q[9];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[0];
h q[9];
h q[2];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
ch q[2], q[9];
cy q[0], q[1];
ry(1.5707963267948966) q[3];
h q[1];
u1(1.5707963267948966) q[7];
t q[9];
crz(1.5707963267948966) q[1], q[3];
h q[4];
ry(1.5707963267948966) q[6];
t q[8];
tdg q[9];
rx(1.5707963267948966) q[6];
rxx(0) q[2], q[8];
cz q[6], q[2];
cu1(1.5707963267948966) q[3], q[7];
tdg q[1];
tdg q[7];
cu3(1.5707963267948966, 0, 0) q[1], q[3];
sdg q[2];
t q[8];
tdg q[2];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[5];
sdg q[5];
tdg q[9];
cu3(1.5707963267948966, 0, 0) q[3], q[1];
rxx(0) q[7], q[9];
x q[3];
t q[4];
ryy(1.5707963267948966) q[9], q[7];
rx(1.5707963267948966) q[0];
sdg q[7];
rx(1.5707963267948966) q[4];
x q[2];
cz q[5], q[2];
crz(1.5707963267948966) q[9], q[4];
rx(1.5707963267948966) q[8];
sdg q[3];
cu3(1.5707963267948966, 0, 0) q[9], q[3];
id q[1];
rz(1.5707963267948966) q[0];
ch q[6], q[4];
u3(0, 0, 1.5707963267948966) q[1];
x q[7];
h q[8];
u3(0, 0, 1.5707963267948966) q[9];
tdg q[3];
sdg q[8];
rx(1.5707963267948966) q[2];
p(0) q[5];
id q[4];
cy q[1], q[0];
tdg q[0];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[9];
s q[2];
cu3(1.5707963267948966, 0, 0) q[9], q[3];
h q[1];
u3(0, 0, 1.5707963267948966) q[7];
id q[7];
h q[7];
ry(1.5707963267948966) q[6];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[7];
cx q[3], q[1];
p(0) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
s q[3];
cu3(1.5707963267948966, 0, 0) q[3], q[7];
rx(1.5707963267948966) q[5];
sdg q[3];
rxx(0) q[1], q[3];
sdg q[3];
cu1(1.5707963267948966) q[9], q[4];
sdg q[5];
rz(1.5707963267948966) q[7];
h q[7];
rx(1.5707963267948966) q[5];
rzz(1.5707963267948966) q[6], q[3];
cz q[6], q[4];
s q[5];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[5];
h q[8];
s q[7];
u3(0, 0, 1.5707963267948966) q[4];
t q[2];
t q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rzz(1.5707963267948966) q[4], q[7];
s q[0];
t q[1];
rz(1.5707963267948966) q[4];
id q[9];
ry(1.5707963267948966) q[2];