OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
tdg q[0];
t q[1];
z q[3];
sdg q[5];
s q[1];
t q[4];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[4];
z q[5];
rz(1.5707963267948966) q[1];
id q[5];
u1(1.5707963267948966) q[5];
t q[4];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[4];
sdg q[1];
sdg q[4];
rz(1.5707963267948966) q[2];
s q[0];
id q[5];
tdg q[4];
tdg q[1];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[2];
s q[0];
u1(1.5707963267948966) q[5];
s q[4];
z q[1];
id q[5];
id q[1];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[0];
s q[2];
z q[4];
s q[5];
