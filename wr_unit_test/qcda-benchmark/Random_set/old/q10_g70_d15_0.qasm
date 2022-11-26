OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
s q[6];
rzz(1.5707963267948966) q[5], q[4];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
ryy(1.5707963267948966) q[2], q[3];
cz q[5], q[8];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
cu1(1.5707963267948966) q[0], q[8];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[6];
id q[8];
t q[0];
id q[3];
id q[3];
rzz(1.5707963267948966) q[4], q[5];
t q[6];
u3(0, 0, 1.5707963267948966) q[7];
t q[8];
ry(1.5707963267948966) q[8];
t q[3];
cu1(1.5707963267948966) q[0], q[2];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
x q[8];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[5];
x q[0];
x q[9];
x q[6];
rzz(1.5707963267948966) q[6], q[2];
u3(0, 0, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[2];
id q[6];
cu3(1.5707963267948966, 0, 0) q[5], q[0];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[4];
t q[0];
id q[1];
h q[1];
p(0) q[6];
u1(1.5707963267948966) q[0];
swap q[4], q[1];
sdg q[0];
ryy(1.5707963267948966) q[6], q[3];
x q[4];
swap q[5], q[3];
u1(1.5707963267948966) q[9];
cx q[9], q[1];
x q[4];
cz q[6], q[7];
ryy(1.5707963267948966) q[9], q[8];
rx(1.5707963267948966) q[3];
id q[9];
t q[1];
id q[0];
rxx(0) q[1], q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cu3(1.5707963267948966, 0, 0) q[6], q[3];
rx(1.5707963267948966) q[4];
cu1(1.5707963267948966) q[9], q[7];
id q[8];
rz(1.5707963267948966) q[2];
ch q[7], q[2];
rz(1.5707963267948966) q[4];
rxx(0) q[2], q[4];