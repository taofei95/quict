OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(1.5707963267948966) q[1];
id q[0];
tdg q[1];
id q[0];
tdg q[0];
tdg q[0];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
sdg q[0];
tdg q[0];
rz(1.5707963267948966) q[1];
tdg q[0];
z q[0];
u1(1.5707963267948966) q[1];
tdg q[0];
u1(1.5707963267948966) q[1];
tdg q[0];
sdg q[0];
rz(1.5707963267948966) q[0];
s q[1];
u1(1.5707963267948966) q[1];
sdg q[0];
sdg q[0];
z q[0];
t q[1];
z q[0];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[0];
t q[0];
id q[1];
s q[0];
rz(1.5707963267948966) q[0];
sdg q[0];
t q[0];
s q[0];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
tdg q[1];
s q[1];
tdg q[0];
s q[1];
s q[0];
tdg q[0];
rz(1.5707963267948966) q[1];
sdg q[1];
rz(1.5707963267948966) q[1];
t q[0];
tdg q[1];
tdg q[1];
u1(1.5707963267948966) q[0];
z q[1];
s q[0];
