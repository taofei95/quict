OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
tdg q[5];
tdg q[5];
s q[6];
tdg q[5];
t q[0];
rz(1.5707963267948966) q[3];
sdg q[3];
sdg q[5];
z q[2];
sdg q[0];
sdg q[6];
s q[0];
rz(1.5707963267948966) q[3];
z q[4];
t q[6];
z q[6];
z q[3];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[6];
sdg q[5];
z q[2];
z q[5];
s q[5];
z q[1];
id q[5];
s q[3];
id q[6];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
t q[2];
u1(1.5707963267948966) q[2];
id q[5];
s q[6];
s q[4];
tdg q[2];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[1];
sdg q[6];
sdg q[5];
t q[2];
rz(1.5707963267948966) q[5];