OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
t q[2];
u1(1.5707963267948966) q[2];
t q[2];
t q[1];
id q[1];
id q[2];
z q[1];
t q[1];
sdg q[1];
rz(1.5707963267948966) q[0];
tdg q[2];
tdg q[1];
id q[1];
sdg q[0];
u1(1.5707963267948966) q[0];
id q[1];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[2];
t q[2];
u1(1.5707963267948966) q[0];
z q[1];
sdg q[2];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
t q[0];
t q[0];
rz(1.5707963267948966) q[0];
tdg q[2];
s q[1];
rz(1.5707963267948966) q[1];
t q[2];
t q[0];
u1(1.5707963267948966) q[2];
t q[1];
id q[2];
t q[1];
z q[1];
t q[2];
s q[2];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
s q[1];
sdg q[2];
t q[0];
id q[2];
z q[1];
t q[0];
id q[2];
tdg q[2];
sdg q[2];
id q[1];
id q[0];
u1(1.5707963267948966) q[1];
tdg q[1];
z q[0];
tdg q[1];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[2];
