OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(1.5707963267948966) q[1];
s q[0];
u3(0, 0, 1.5707963267948966) q[1];
y q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
x q[1];
tdg q[0];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
t q[0];
z q[0];
rz(1.5707963267948966) q[1];
s q[1];
z q[1];
u3(0, 0, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[0];
