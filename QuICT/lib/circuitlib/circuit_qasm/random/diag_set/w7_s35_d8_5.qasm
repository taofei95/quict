OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
sdg q[6];
tdg q[2];
u1(1.5707963267948966) q[1];
sdg q[2];
t q[4];
tdg q[6];
rz(1.5707963267948966) q[1];
t q[3];
id q[0];
u1(1.5707963267948966) q[5];
sdg q[6];
t q[0];
rz(1.5707963267948966) q[6];
sdg q[6];
t q[3];
sdg q[4];
tdg q[1];
s q[5];
sdg q[2];
rz(1.5707963267948966) q[0];
t q[5];
sdg q[4];
sdg q[6];
u1(1.5707963267948966) q[5];
tdg q[4];
tdg q[5];
rz(1.5707963267948966) q[6];
sdg q[4];
sdg q[1];
s q[4];
id q[0];
sdg q[4];
z q[1];
z q[1];
u1(1.5707963267948966) q[4];
