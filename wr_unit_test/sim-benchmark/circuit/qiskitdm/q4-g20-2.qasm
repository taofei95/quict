OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(1.5707963267948966) q[2];
tdg q[1];
sdg q[0];
id q[2];
sdg q[1];
sdg q[1];
u1(1.5707963267948966) q[1];
t q[1];
sdg q[3];
rxx(0) q[2], q[3];
p(0) q[0];
tdg q[2];
tdg q[3];
rxx(0) q[3], q[1];
s q[2];
ry(1.5707963267948966) q[0];
cu1(1.5707963267948966) q[1], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
p(0) q[3];