OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
sdg q[1];
h q[6];
rzz(1.5707963267948966) q[7], q[8];
rz(1.5707963267948966) q[5];
h q[0];
ch q[2], q[1];
tdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[2];
swap q[7], q[3];
s q[9];
swap q[8], q[2];
cy q[5], q[1];
rx(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[0];
ch q[7], q[4];
ry(1.5707963267948966) q[8];
p(0) q[3];
rx(1.5707963267948966) q[7];
id q[6];
tdg q[2];
cx q[1], q[9];
rxx(0) q[5], q[7];
s q[8];
rz(1.5707963267948966) q[8];
p(0) q[1];
id q[6];
s q[1];
cx q[1], q[9];
rx(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[2];
x q[1];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[9];
tdg q[5];
sdg q[9];
sdg q[0];
u1(1.5707963267948966) q[4];
ch q[7], q[3];
id q[3];
ry(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
id q[3];
ch q[6], q[1];
rx(1.5707963267948966) q[2];
t q[9];
rx(1.5707963267948966) q[3];