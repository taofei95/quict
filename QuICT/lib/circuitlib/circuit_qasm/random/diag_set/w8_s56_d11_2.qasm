OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
z q[6];
t q[4];
tdg q[2];
tdg q[3];
z q[3];
u1(1.5707963267948966) q[6];
t q[6];
z q[4];
t q[6];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
s q[0];
rz(1.5707963267948966) q[6];
z q[0];
z q[1];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[7];
id q[4];
sdg q[0];
z q[1];
t q[7];
s q[7];
s q[2];
u1(1.5707963267948966) q[0];
id q[2];
t q[1];
s q[7];
t q[6];
tdg q[5];
rz(1.5707963267948966) q[3];
t q[7];
z q[7];
tdg q[6];
t q[0];
rz(1.5707963267948966) q[5];
id q[0];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[7];
t q[0];
s q[0];
tdg q[5];
sdg q[1];
s q[7];
tdg q[6];
s q[5];
z q[3];
tdg q[2];
t q[2];
rz(1.5707963267948966) q[1];
tdg q[7];
u1(1.5707963267948966) q[4];
sdg q[1];
s q[1];
s q[3];
tdg q[7];
z q[5];
