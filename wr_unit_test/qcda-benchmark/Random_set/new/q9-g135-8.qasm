OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
cz q[3], q[8];
sdg q[7];
ryy(1.5707963267948966) q[7], q[3];
p(0) q[4];
swap q[2], q[6];
h q[6];
rzz(1.5707963267948966) q[3], q[2];
swap q[3], q[2];
x q[2];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[6];
sdg q[8];
s q[4];
u3(0, 0, 1.5707963267948966) q[6];
rxx(0) q[4], q[3];
cx q[8], q[2];
tdg q[5];
rx(1.5707963267948966) q[2];
p(0) q[7];
ryy(1.5707963267948966) q[7], q[0];
rzz(1.5707963267948966) q[7], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[1];
h q[8];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[8];
t q[7];
u1(1.5707963267948966) q[3];
s q[2];
rx(1.5707963267948966) q[2];
h q[7];
p(0) q[7];
sdg q[7];
h q[7];
rz(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[4];
t q[0];
crz(1.5707963267948966) q[2], q[7];
h q[3];
h q[3];
p(0) q[4];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[5];
t q[5];
tdg q[0];
t q[0];
p(0) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
tdg q[2];
rz(1.5707963267948966) q[1];
h q[4];
cz q[5], q[0];
crz(1.5707963267948966) q[3], q[1];
sdg q[0];
rxx(0) q[4], q[3];
id q[3];
x q[3];
rx(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[7];
p(0) q[0];
t q[3];
rxx(0) q[4], q[6];
u3(0, 0, 1.5707963267948966) q[7];
x q[4];
id q[5];
rz(1.5707963267948966) q[1];
x q[1];
sdg q[2];
tdg q[1];
rx(1.5707963267948966) q[4];
u1(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[3], q[2];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[8];
h q[1];
p(0) q[6];
sdg q[6];
s q[6];
p(0) q[1];
cy q[8], q[7];
cu1(1.5707963267948966) q[6], q[3];
cx q[4], q[2];
tdg q[0];
rxx(0) q[5], q[8];
cy q[5], q[3];
h q[1];
cz q[0], q[4];
sdg q[0];
cz q[2], q[4];
x q[1];
id q[2];
p(0) q[0];
rxx(0) q[1], q[2];
tdg q[1];
rx(1.5707963267948966) q[0];
rzz(1.5707963267948966) q[6], q[2];
id q[2];
ryy(1.5707963267948966) q[7], q[1];
tdg q[7];
rx(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[8];
tdg q[0];
sdg q[1];
x q[2];
ch q[1], q[0];
h q[8];
u2(1.5707963267948966, 1.5707963267948966) q[0];
swap q[6], q[1];
u1(1.5707963267948966) q[6];
rzz(1.5707963267948966) q[8], q[5];
t q[5];
h q[2];
crz(1.5707963267948966) q[1], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
p(0) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rzz(1.5707963267948966) q[1], q[4];
p(0) q[2];
u1(1.5707963267948966) q[5];
s q[3];
u3(0, 0, 1.5707963267948966) q[1];
s q[6];
cz q[0], q[2];
rx(1.5707963267948966) q[4];
x q[3];
ry(1.5707963267948966) q[8];
sdg q[3];
s q[1];
u1(1.5707963267948966) q[3];
tdg q[8];
h q[8];