OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
s q[3];
u1(1.5707963267948966) q[0];
sdg q[0];
rz(1.5707963267948966) q[2];
z q[2];
u1(1.5707963267948966) q[1];
s q[3];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
z q[0];
z q[3];
y q[1];
y q[3];
rx(1.5707963267948966) q[2];
y q[2];
ry(1.5707963267948966) q[1];
tdg q[1];
