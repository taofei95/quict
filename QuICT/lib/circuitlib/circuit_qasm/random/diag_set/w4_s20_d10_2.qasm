OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
sdg q[2];
s q[2];
rz(1.5707963267948966) q[2];
sdg q[1];
s q[1];
id q[2];
tdg q[3];
z q[1];
s q[1];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[2];
s q[3];
sdg q[1];
z q[2];
id q[2];
rz(1.5707963267948966) q[2];
tdg q[2];
id q[0];
tdg q[3];
rz(1.5707963267948966) q[2];
