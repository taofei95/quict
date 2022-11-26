OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cu1(1.5707963267948966) q[4], q[5];
ch q[5], q[0];
id q[4];
cz q[5], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
id q[1];
s q[0];
cu1(1.5707963267948966) q[2], q[3];
p(0) q[1];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
h q[3];
rxx(0) q[4], q[3];
cu3(1.5707963267948966, 0, 0) q[4], q[3];
u1(1.5707963267948966) q[2];
cz q[4], q[1];
h q[4];
cu1(1.5707963267948966) q[2], q[5];
crz(1.5707963267948966) q[5], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
cu3(1.5707963267948966, 0, 0) q[5], q[0];
h q[4];
p(0) q[5];
u3(0, 0, 1.5707963267948966) q[2];
swap q[2], q[1];
h q[0];
rz(1.5707963267948966) q[5];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[0];
ch q[5], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[3];
swap q[4], q[5];
rx(1.5707963267948966) q[4];
id q[0];
sdg q[2];
swap q[3], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[0];
h q[2];
cy q[1], q[0];
rx(1.5707963267948966) q[5];
swap q[0], q[3];
h q[2];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
s q[2];
s q[1];
cx q[2], q[0];
rxx(0) q[4], q[2];
rxx(0) q[3], q[0];
rx(1.5707963267948966) q[5];
p(0) q[2];
sdg q[5];
h q[3];
cu1(1.5707963267948966) q[1], q[3];
ry(1.5707963267948966) q[1];
p(0) q[2];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[3];
rxx(0) q[5], q[4];
cu1(1.5707963267948966) q[0], q[3];
rx(1.5707963267948966) q[5];
p(0) q[5];
rz(1.5707963267948966) q[1];
tdg q[5];
sdg q[4];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
h q[4];
id q[1];