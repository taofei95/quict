OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
s q[5];
p(0) q[1];
cu1(1.5707963267948966) q[0], q[3];
cx q[5], q[1];
h q[2];
rz(1.5707963267948966) q[2];
rxx(0) q[4], q[2];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
sdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
h q[4];
id q[0];
x q[2];
t q[0];
rxx(0) q[2], q[3];
id q[4];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
crz(1.5707963267948966) q[0], q[4];
sdg q[4];
sdg q[3];
sdg q[5];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[2];
x q[4];
rx(1.5707963267948966) q[3];
ch q[0], q[1];
rz(1.5707963267948966) q[4];
cz q[2], q[4];
tdg q[3];
ry(1.5707963267948966) q[3];
id q[1];
h q[4];
tdg q[0];
id q[4];
x q[0];
p(0) q[5];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
id q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
tdg q[3];
h q[3];
cy q[2], q[4];
swap q[5], q[4];
ryy(1.5707963267948966) q[3], q[5];
ch q[1], q[4];
s q[5];
u1(1.5707963267948966) q[4];
s q[4];
ry(1.5707963267948966) q[4];
p(0) q[2];
tdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
sdg q[0];
u3(0, 0, 1.5707963267948966) q[1];
sdg q[5];
ch q[5], q[1];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[4];
rzz(1.5707963267948966) q[3], q[4];
cz q[0], q[2];
t q[1];
u1(1.5707963267948966) q[5];
swap q[1], q[2];
p(0) q[4];
swap q[2], q[5];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
ryy(1.5707963267948966) q[2], q[5];
x q[2];
sdg q[2];
ryy(1.5707963267948966) q[4], q[5];
ryy(1.5707963267948966) q[0], q[2];
t q[5];