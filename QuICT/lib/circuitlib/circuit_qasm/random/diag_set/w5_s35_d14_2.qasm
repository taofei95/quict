OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(1.5707963267948966) q[0];
z q[3];
z q[3];
tdg q[3];
id q[0];
s q[0];
s q[3];
rz(1.5707963267948966) q[2];
s q[1];
rz(1.5707963267948966) q[0];
z q[2];
id q[4];
sdg q[4];
tdg q[2];
z q[0];
t q[3];
tdg q[0];
t q[2];
t q[3];
id q[0];
z q[1];
sdg q[0];
rz(1.5707963267948966) q[0];
tdg q[2];
tdg q[3];
id q[0];
sdg q[0];
u1(1.5707963267948966) q[3];
z q[2];
rz(1.5707963267948966) q[0];
z q[1];
rz(1.5707963267948966) q[0];
t q[0];
u1(1.5707963267948966) q[2];
id q[2];
