OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
sdg q[5];
s q[2];
rx(1.5707963267948966) q[5];
h q[4];
cx q[1], q[3];
rz(1.5707963267948966) q[0];
ch q[1], q[3];
id q[1];
sdg q[2];
t q[3];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[5];
cx q[0], q[1];
rx(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
crz(1.5707963267948966) q[5], q[2];
s q[2];
t q[0];
u1(1.5707963267948966) q[1];
rxx(0) q[5], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
tdg q[2];
swap q[2], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
p(0) q[1];
h q[5];
ch q[3], q[1];
ry(1.5707963267948966) q[2];
p(0) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
p(0) q[0];
u3(0, 0, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
swap q[0], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
s q[1];
rz(1.5707963267948966) q[1];
ch q[0], q[2];
s q[4];
h q[0];
rx(1.5707963267948966) q[0];
id q[1];
id q[3];
sdg q[4];
rzz(1.5707963267948966) q[3], q[0];
tdg q[3];
t q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
cu1(1.5707963267948966) q[3], q[0];
tdg q[1];
cz q[5], q[1];
id q[0];
tdg q[5];
u3(0, 0, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
cu1(1.5707963267948966) q[2], q[5];
rzz(1.5707963267948966) q[4], q[3];
id q[4];
p(0) q[0];
tdg q[2];
swap q[4], q[2];
tdg q[3];
rz(1.5707963267948966) q[5];
t q[0];
u3(0, 0, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[5];
tdg q[2];
sdg q[0];
cx q[1], q[2];
s q[3];
ch q[5], q[1];
tdg q[5];
rzz(1.5707963267948966) q[0], q[3];
swap q[2], q[0];
rzz(1.5707963267948966) q[5], q[1];
t q[4];
u1(1.5707963267948966) q[5];
s q[2];
sdg q[3];
cu3(1.5707963267948966, 0, 0) q[0], q[2];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[5];