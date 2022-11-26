OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
crz(1.5707963267948966) q[3], q[1];
tdg q[0];
x q[0];
h q[2];
cu1(1.5707963267948966) q[3], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];
t q[3];
tdg q[1];
ry(1.5707963267948966) q[3];
swap q[0], q[1];
ry(1.5707963267948966) q[4];
x q[2];
ry(1.5707963267948966) q[0];
p(0) q[2];
id q[2];
sdg q[1];
t q[2];
p(0) q[4];
u1(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
s q[2];
s q[2];