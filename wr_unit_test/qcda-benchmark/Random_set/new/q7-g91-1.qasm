OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[4];
id q[2];
ch q[1], q[3];
id q[2];
id q[6];
h q[4];
cy q[4], q[2];
p(0) q[6];
sdg q[4];
rxx(0) q[1], q[0];
sdg q[3];
u3(0, 0, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[6];
cy q[4], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[2];
s q[4];
tdg q[3];
id q[4];
s q[3];
p(0) q[6];
sdg q[0];
rzz(1.5707963267948966) q[2], q[4];
cz q[1], q[2];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[3];
id q[2];
s q[1];
u1(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[2], q[1];
ryy(1.5707963267948966) q[6], q[3];
p(0) q[0];
id q[5];
p(0) q[2];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ryy(1.5707963267948966) q[6], q[0];
sdg q[0];
cy q[5], q[0];
id q[0];
crz(1.5707963267948966) q[0], q[2];
ry(1.5707963267948966) q[1];
cx q[0], q[3];
ry(1.5707963267948966) q[1];
crz(1.5707963267948966) q[5], q[6];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
t q[5];
sdg q[5];
rz(1.5707963267948966) q[6];
rxx(0) q[3], q[1];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
h q[5];
rx(1.5707963267948966) q[6];
sdg q[5];
tdg q[5];
p(0) q[4];
id q[6];
sdg q[0];
tdg q[2];
ryy(1.5707963267948966) q[2], q[4];
tdg q[4];
id q[2];
t q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[6];
t q[0];
cx q[4], q[6];
rz(1.5707963267948966) q[3];
x q[6];
ry(1.5707963267948966) q[3];
t q[0];
u3(0, 0, 1.5707963267948966) q[4];
crz(1.5707963267948966) q[3], q[6];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[1];
cy q[0], q[6];
x q[1];
sdg q[3];
x q[4];
sdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
tdg q[1];
tdg q[2];
ryy(1.5707963267948966) q[1], q[3];
cx q[0], q[5];