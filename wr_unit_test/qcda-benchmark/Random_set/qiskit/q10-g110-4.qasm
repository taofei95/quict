OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
t q[3];
cz q[8], q[7];
tdg q[1];
t q[3];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[7];
tdg q[1];
rz(1.5707963267948966) q[5];
s q[4];
rx(1.5707963267948966) q[6];
crz(1.5707963267948966) q[5], q[4];
rz(1.5707963267948966) q[6];
id q[9];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
h q[6];
h q[9];
ch q[8], q[1];
tdg q[3];
cx q[0], q[8];
u3(0, 0, 1.5707963267948966) q[7];
cz q[2], q[8];
sdg q[4];
u1(1.5707963267948966) q[3];
cx q[5], q[9];
t q[0];
u3(0, 0, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[2];
tdg q[5];
p(0) q[9];
swap q[0], q[5];
h q[6];
t q[2];
u1(1.5707963267948966) q[0];
s q[9];
sdg q[1];
id q[3];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
s q[5];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[6];
t q[4];
u3(0, 0, 1.5707963267948966) q[3];
p(0) q[6];
crz(1.5707963267948966) q[6], q[4];
rx(1.5707963267948966) q[9];
p(0) q[0];
t q[2];
p(0) q[4];
cx q[8], q[9];
p(0) q[1];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[9];
tdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[0];
tdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[0];
ch q[2], q[0];
tdg q[1];
swap q[2], q[5];
sdg q[2];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[9];
sdg q[1];
rx(1.5707963267948966) q[4];
s q[6];
sdg q[5];
rzz(1.5707963267948966) q[0], q[2];
u3(0, 0, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[7];
rx(1.5707963267948966) q[7];
id q[6];
p(0) q[1];
sdg q[3];
id q[2];
swap q[6], q[1];
h q[4];
p(0) q[5];
p(0) q[8];
tdg q[8];
s q[8];
ry(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[3];
crz(1.5707963267948966) q[5], q[2];
tdg q[8];
rxx(0) q[3], q[4];
crz(1.5707963267948966) q[8], q[2];
cx q[7], q[5];
s q[2];
h q[2];
u1(1.5707963267948966) q[1];
h q[9];
tdg q[2];
h q[1];
u1(1.5707963267948966) q[3];
p(0) q[4];
u1(1.5707963267948966) q[2];