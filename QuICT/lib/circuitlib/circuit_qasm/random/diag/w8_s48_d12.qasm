OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
sdg q[4];
id q[1];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
id q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
t q[7];
sdg q[5];
sdg q[7];
s q[7];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
sdg q[4];
tdg q[1];
id q[4];
z q[1];
u1(1.5707963267948966) q[1];
tdg q[1];
t q[4];
z q[2];
t q[3];
z q[4];
sdg q[6];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[7];
t q[4];
rz(1.5707963267948966) q[5];
sdg q[7];
sdg q[5];
id q[2];
u1(1.5707963267948966) q[0];
z q[4];
u1(1.5707963267948966) q[4];
t q[4];
t q[6];
t q[7];
id q[7];
u1(1.5707963267948966) q[1];
sdg q[1];
id q[7];
tdg q[0];
id q[1];
rz(1.5707963267948966) q[3];
s q[6];
id q[4];
