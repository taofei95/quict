OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rx(1.5707963267948966) q[1];
id q[2];
ry(1.5707963267948966) q[2];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
p(0) q[0];
cx q[4], q[0];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
tdg q[2];
x q[3];
swap q[0], q[2];
rz(1.5707963267948966) q[3];
tdg q[0];
cu3(1.5707963267948966, 0, 0) q[3], q[4];
rzz(1.5707963267948966) q[4], q[0];
rz(1.5707963267948966) q[0];
t q[1];
tdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[4];
sdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
cu3(1.5707963267948966, 0, 0) q[2], q[4];
tdg q[4];
id q[1];
rxx(0) q[4], q[2];
x q[0];
x q[0];
id q[1];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
cx q[4], q[0];
h q[3];
p(0) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
sdg q[1];
tdg q[4];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
cz q[4], q[1];
tdg q[1];
cx q[1], q[3];
swap q[4], q[2];
rx(1.5707963267948966) q[3];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
sdg q[3];
cu1(1.5707963267948966) q[4], q[3];
rx(1.5707963267948966) q[0];
h q[4];
cz q[4], q[3];
t q[3];