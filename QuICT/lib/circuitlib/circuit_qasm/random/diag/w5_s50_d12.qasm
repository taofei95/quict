OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(1.5707963267948966) q[4];
id q[3];
id q[2];
z q[2];
u1(1.5707963267948966) q[3];
id q[0];
tdg q[0];
s q[2];
u1(1.5707963267948966) q[3];
sdg q[3];
t q[3];
u1(1.5707963267948966) q[3];
tdg q[2];
s q[0];
t q[0];
s q[0];
sdg q[3];
z q[4];
t q[4];
tdg q[4];
z q[0];
u1(1.5707963267948966) q[1];
tdg q[0];
z q[4];
id q[1];
sdg q[4];
z q[2];
id q[1];
z q[2];
t q[2];
id q[2];
rz(1.5707963267948966) q[4];
id q[1];
u1(1.5707963267948966) q[4];
z q[2];
u1(1.5707963267948966) q[1];
t q[4];
sdg q[4];
id q[3];
t q[0];
sdg q[4];
sdg q[1];
id q[1];
id q[4];
sdg q[1];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
t q[0];
sdg q[3];
