OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
p(0) q[6];
cu1(1.5707963267948966) q[8], q[3];
rx(1.5707963267948966) q[5];
id q[1];
tdg q[3];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[0];
cz q[0], q[5];
cx q[2], q[8];
t q[4];
id q[3];
cx q[7], q[1];
rxx(0) q[8], q[1];
rz(1.5707963267948966) q[7];
cx q[3], q[1];
cx q[8], q[7];
rzz(1.5707963267948966) q[5], q[3];
rzz(1.5707963267948966) q[4], q[7];
cy q[7], q[6];
rzz(1.5707963267948966) q[7], q[1];
rxx(0) q[1], q[7];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[8];
u1(1.5707963267948966) q[8];
tdg q[7];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
t q[5];
t q[5];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
cx q[8], q[2];
s q[5];
u1(1.5707963267948966) q[6];
rx(1.5707963267948966) q[0];
h q[5];
s q[2];
cx q[8], q[7];
t q[1];
u3(0, 0, 1.5707963267948966) q[3];
h q[5];
rz(1.5707963267948966) q[0];
h q[5];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[8];
p(0) q[6];
rz(1.5707963267948966) q[2];
id q[5];
u2(1.5707963267948966, 1.5707963267948966) q[8];
cx q[2], q[1];
rxx(0) q[6], q[3];
rzz(1.5707963267948966) q[1], q[5];
h q[4];
u1(1.5707963267948966) q[6];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
swap q[4], q[0];
p(0) q[5];
tdg q[2];
u1(1.5707963267948966) q[7];
p(0) q[4];