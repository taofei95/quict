OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
sdg q[1];
rz(1.5707963267948966) q[0];
z q[1];
s q[5];
s q[4];
z q[4];
id q[2];
sdg q[1];
tdg q[5];
u1(1.5707963267948966) q[2];
id q[4];
sdg q[0];
s q[2];
sdg q[5];
u1(1.5707963267948966) q[5];
sdg q[5];
tdg q[3];
u1(1.5707963267948966) q[5];
sdg q[1];
s q[2];
u1(1.5707963267948966) q[0];
id q[0];
tdg q[2];
s q[0];
id q[2];
u1(1.5707963267948966) q[4];
z q[1];
id q[1];
tdg q[5];
t q[1];
tdg q[1];
t q[1];
tdg q[4];
sdg q[5];
s q[2];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[0];
id q[0];
s q[4];
s q[0];
rz(1.5707963267948966) q[5];
id q[2];
tdg q[4];
z q[3];
u1(1.5707963267948966) q[3];
z q[1];
sdg q[0];
z q[4];
id q[2];
id q[5];
tdg q[1];
t q[4];
tdg q[2];
tdg q[5];
z q[0];
id q[0];
id q[0];
tdg q[2];
sdg q[4];
sdg q[2];