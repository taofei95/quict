OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
z q[1];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
t q[0];
u1(1.5707963267948966) q[0];
t q[1];
rz(1.5707963267948966) q[0];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[1];
