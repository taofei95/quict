OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
sdg q[1];
ry(1.5707963267948966) q[1];
t q[2];
tdg q[2];
s q[2];
rz(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[3];
sdg q[0];
s q[3];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[3];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
sdg q[2];
u1(1.5707963267948966) q[2];
sdg q[2];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
sdg q[3];
h q[3];
h q[0];
t q[1];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[0];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
