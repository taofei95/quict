OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
tdg q[2];
sdg q[3];
sdg q[2];
rz(1.5707963267948966) q[2];
sdg q[0];
z q[1];
tdg q[4];
z q[1];
u1(1.5707963267948966) q[0];
s q[2];
s q[4];
sdg q[2];
sdg q[2];
id q[4];
u1(1.5707963267948966) q[0];
s q[4];
sdg q[3];
s q[2];
t q[2];
sdg q[2];
s q[3];
z q[2];
s q[4];
z q[2];
id q[2];
s q[2];
rz(1.5707963267948966) q[3];
id q[0];
id q[1];
u1(1.5707963267948966) q[0];
t q[3];
s q[4];
rz(1.5707963267948966) q[0];
s q[1];
z q[3];
t q[0];
rz(1.5707963267948966) q[0];
t q[2];
tdg q[4];
z q[3];
rz(1.5707963267948966) q[3];
sdg q[4];
id q[3];
tdg q[1];
tdg q[1];
