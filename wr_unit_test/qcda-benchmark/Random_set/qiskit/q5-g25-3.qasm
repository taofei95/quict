OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
swap q[2], q[3];
u3(0, 0, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[2];
t q[4];
swap q[2], q[3];
u3(0, 0, 1.5707963267948966) q[1];
id q[1];
t q[3];
t q[0];
u3(0, 0, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
cz q[4], q[1];
id q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
swap q[2], q[0];
ch q[3], q[1];
tdg q[2];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[4];
sdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
p(0) q[0];
u1(1.5707963267948966) q[3];