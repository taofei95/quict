OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
sdg q[9];
id q[5];
tdg q[1];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[9];
z q[1];
u1(1.5707963267948966) q[2];
s q[6];
s q[1];
t q[9];
t q[6];
t q[0];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
tdg q[8];
id q[9];
id q[5];
s q[3];
sdg q[1];
z q[3];
t q[7];
sdg q[1];
t q[8];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[8];
u1(1.5707963267948966) q[4];
id q[7];
tdg q[6];
rz(1.5707963267948966) q[6];
s q[5];
id q[4];
id q[3];
sdg q[1];
t q[6];
id q[4];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[0];
sdg q[7];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
t q[9];
z q[4];
z q[7];
u1(1.5707963267948966) q[9];
rz(1.5707963267948966) q[2];
z q[9];
rz(1.5707963267948966) q[7];
sdg q[9];
id q[9];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[3];
tdg q[5];
tdg q[2];
sdg q[9];
u1(1.5707963267948966) q[2];
id q[4];
z q[4];
id q[9];
t q[7];
t q[8];
tdg q[5];
sdg q[6];
rz(1.5707963267948966) q[8];
t q[4];
tdg q[0];
u1(1.5707963267948966) q[2];
s q[6];
s q[7];
sdg q[4];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[6];
sdg q[6];
rz(1.5707963267948966) q[3];
id q[0];
id q[4];
rz(1.5707963267948966) q[0];
sdg q[6];
sdg q[7];
z q[1];
s q[4];
s q[3];
tdg q[4];
z q[3];
t q[7];
id q[0];
z q[4];
t q[0];
t q[8];
id q[0];
rz(1.5707963267948966) q[2];
sdg q[4];
z q[6];
sdg q[7];
u1(1.5707963267948966) q[4];
tdg q[2];
u1(1.5707963267948966) q[3];
s q[7];
id q[6];
t q[4];
