OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
sdg q[0];
s q[2];
t q[2];
id q[1];
s q[3];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
s q[0];
sdg q[0];
sdg q[4];
rz(1.5707963267948966) q[4];
tdg q[0];
t q[4];
s q[1];
t q[4];
u1(1.5707963267948966) q[2];
z q[2];
t q[4];
t q[0];
s q[3];
z q[1];
z q[4];
rz(1.5707963267948966) q[3];
sdg q[2];
sdg q[1];
