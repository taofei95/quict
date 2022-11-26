OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
ch q[1], q[2];
swap q[6], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
id q[3];
rz(1.5707963267948966) q[5];
crz(1.5707963267948966) q[5], q[6];
p(0) q[2];
tdg q[0];
s q[6];
tdg q[5];
s q[3];
h q[1];
h q[4];
rz(1.5707963267948966) q[6];
cx q[1], q[5];
ry(1.5707963267948966) q[4];
swap q[3], q[4];
s q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
cz q[3], q[4];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
tdg q[6];
cy q[1], q[5];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[1];
s q[6];