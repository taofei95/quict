OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
id q[5];
rz(1.5707963267948966) q[2];
t q[3];
tdg q[3];
sdg q[5];
id q[4];
z q[1];
tdg q[1];
sdg q[4];
s q[0];
z q[0];
t q[0];
z q[0];
sdg q[7];
s q[4];
t q[0];
tdg q[2];
t q[2];
rz(1.5707963267948966) q[5];
sdg q[6];
s q[0];
s q[7];
u1(1.5707963267948966) q[2];
sdg q[2];
u1(1.5707963267948966) q[1];
sdg q[7];
t q[1];
rz(1.5707963267948966) q[1];
s q[2];
t q[7];
sdg q[0];
t q[5];
s q[6];
u1(1.5707963267948966) q[4];
s q[5];
sdg q[0];
t q[6];
z q[4];
id q[0];
rz(1.5707963267948966) q[1];
s q[7];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[5];
id q[5];
sdg q[0];
s q[5];
s q[4];
s q[7];
