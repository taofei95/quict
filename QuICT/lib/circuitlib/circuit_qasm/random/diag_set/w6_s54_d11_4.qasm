OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
tdg q[4];
sdg q[0];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[5];
tdg q[2];
t q[0];
s q[1];
t q[1];
u1(1.5707963267948966) q[4];
sdg q[0];
id q[5];
tdg q[5];
sdg q[3];
t q[5];
tdg q[1];
u1(1.5707963267948966) q[3];
id q[2];
t q[5];
sdg q[3];
tdg q[2];
id q[3];
t q[2];
id q[5];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
s q[3];
u1(1.5707963267948966) q[3];
id q[2];
u1(1.5707963267948966) q[5];
t q[5];
z q[1];
tdg q[3];
z q[1];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];
tdg q[1];
rz(1.5707963267948966) q[0];
sdg q[4];
z q[2];
u1(1.5707963267948966) q[1];
sdg q[4];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
sdg q[4];
z q[2];
u1(1.5707963267948966) q[4];
tdg q[2];
tdg q[0];
tdg q[1];
sdg q[3];
tdg q[2];
u1(1.5707963267948966) q[4];
z q[1];
