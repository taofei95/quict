OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
ry(1.5707963267948966) q[3];
rzz(1.5707963267948966) q[6], q[4];
cu3(1.5707963267948966, 0, 0) q[2], q[6];
ch q[0], q[2];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
id q[4];
ry(1.5707963267948966) q[3];
p(0) q[0];
id q[5];
t q[5];
id q[3];
x q[2];
ry(1.5707963267948966) q[0];
tdg q[1];
u1(1.5707963267948966) q[2];
x q[1];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
sdg q[5];
t q[1];
t q[5];
cy q[2], q[3];
t q[5];
p(0) q[3];
s q[3];
x q[0];
id q[0];
cx q[5], q[6];
s q[4];
rx(1.5707963267948966) q[0];
x q[5];
rx(1.5707963267948966) q[6];
sdg q[2];
cz q[0], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
x q[3];
sdg q[4];
tdg q[0];
tdg q[3];
t q[1];
h q[2];
tdg q[5];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
x q[0];
tdg q[2];
u3(0, 0, 1.5707963267948966) q[4];
t q[2];
cu1(1.5707963267948966) q[1], q[5];
tdg q[2];
tdg q[5];
p(0) q[3];
sdg q[3];
h q[5];
s q[3];
ry(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[2], q[0];
s q[4];
ry(1.5707963267948966) q[1];
swap q[0], q[1];
ry(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[2];
crz(1.5707963267948966) q[2], q[3];
ryy(1.5707963267948966) q[6], q[3];
sdg q[0];
p(0) q[4];
u1(1.5707963267948966) q[3];
rxx(0) q[4], q[2];
sdg q[3];
s q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
swap q[5], q[1];
u3(0, 0, 1.5707963267948966) q[2];
s q[4];
swap q[4], q[2];