OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
id q[0];
s q[1];
s q[1];
u1(1.5707963267948966) q[1];
z q[0];
t q[1];
id q[1];
id q[1];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[1];
