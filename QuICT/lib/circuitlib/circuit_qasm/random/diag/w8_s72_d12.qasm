OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
z q[7];
u1(1.5707963267948966) q[2];
id q[3];
id q[1];
t q[6];
s q[7];
s q[5];
s q[4];
z q[2];
sdg q[0];
s q[6];
z q[2];
rz(1.5707963267948966) q[6];
tdg q[6];
id q[5];
z q[3];
t q[4];
tdg q[4];
id q[4];
tdg q[2];
id q[1];
id q[6];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[7];
sdg q[0];
t q[6];
z q[7];
rz(1.5707963267948966) q[3];
z q[1];
u1(1.5707963267948966) q[7];
s q[6];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[4];
s q[2];
id q[5];
sdg q[4];
t q[5];
z q[5];
sdg q[0];
u1(1.5707963267948966) q[4];
rz(1.5707963267948966) q[7];
id q[2];
id q[4];
id q[6];
id q[2];
tdg q[2];
s q[0];
s q[7];
tdg q[0];
tdg q[1];
s q[5];
rz(1.5707963267948966) q[3];
t q[2];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
z q[6];
tdg q[3];
t q[1];
t q[4];
z q[6];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[6];
tdg q[0];
rz(1.5707963267948966) q[0];
t q[2];
tdg q[6];
z q[3];
rz(1.5707963267948966) q[5];
z q[4];
z q[4];
t q[1];
