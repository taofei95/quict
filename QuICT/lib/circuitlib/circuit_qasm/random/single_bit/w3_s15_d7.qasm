OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
sdg q[2];
z q[2];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
z q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
tdg q[1];
t q[0];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[2];