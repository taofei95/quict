OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
sdg q[1];
rz(1.5707963267948966) q[2];
t q[2];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[2];
y q[2];
t q[1];
sdg q[0];
y q[3];
h q[0];
s q[3];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
y q[2];
tdg q[3];
u1(1.5707963267948966) q[2];
y q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
sdg q[3];
u1(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[2];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[1];
