OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
sdg q[2];
u1(1.5707963267948966) q[2];
t q[0];
id q[0];
sdg q[2];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[3];
tdg q[0];
z q[1];
s q[3];
tdg q[3];
s q[3];
tdg q[0];
t q[3];
z q[2];
s q[0];
sdg q[2];
z q[1];
s q[1];
u1(1.5707963267948966) q[2];
t q[2];
s q[1];
s q[0];
u1(1.5707963267948966) q[2];
id q[0];
z q[1];
u1(1.5707963267948966) q[2];
sdg q[3];
z q[3];
s q[2];
t q[3];
z q[1];