OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
ch q[4], q[7];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[5];
rxx(0) q[5], q[0];
h q[5];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[8];
swap q[3], q[8];
u1(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[6];
rzz(1.5707963267948966) q[6], q[8];
t q[7];
cu1(1.5707963267948966) q[2], q[7];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
p(0) q[5];
rxx(0) q[6], q[4];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
tdg q[5];
x q[0];
p(0) q[8];
sdg q[1];
t q[6];
id q[3];
ryy(1.5707963267948966) q[2], q[4];
s q[0];
sdg q[7];
rz(1.5707963267948966) q[3];
h q[6];
ryy(1.5707963267948966) q[7], q[5];
crz(1.5707963267948966) q[4], q[7];
sdg q[1];
s q[1];
ch q[2], q[7];
cu3(1.5707963267948966, 0, 0) q[0], q[8];
cy q[4], q[0];
sdg q[2];
ry(1.5707963267948966) q[4];
s q[4];
h q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[0];
s q[1];
u1(1.5707963267948966) q[6];
t q[2];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[5];
s q[6];
s q[6];
p(0) q[8];
x q[4];
rxx(0) q[4], q[8];
sdg q[1];
u1(1.5707963267948966) q[0];
h q[8];
ry(1.5707963267948966) q[3];
s q[6];
tdg q[0];
t q[0];