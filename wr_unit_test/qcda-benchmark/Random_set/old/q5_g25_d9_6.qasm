OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
u1(1.5707963267948966) q[3];
ryy(1.5707963267948966) q[3], q[4];
t q[1];
h q[0];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[4];
t q[0];
x q[2];
ch q[0], q[1];
x q[2];
p(0) q[0];
cz q[1], q[0];
cz q[1], q[0];
tdg q[2];
x q[1];
id q[1];
tdg q[2];
u1(1.5707963267948966) q[2];
crz(1.5707963267948966) q[3], q[2];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
x q[4];
h q[1];
rx(1.5707963267948966) q[2];
swap q[0], q[3];