OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
t q[2];
t q[1];
s q[0];
id q[1];
u1(1.5707963267948966) q[2];
t q[2];
sdg q[2];
rz(1.5707963267948966) q[3];
s q[3];
id q[2];
tdg q[1];
rz(1.5707963267948966) q[3];
id q[2];
z q[2];
sdg q[1];
id q[1];
sdg q[2];
rz(1.5707963267948966) q[2];
tdg q[0];
u1(1.5707963267948966) q[1];
t q[2];
sdg q[2];
