OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg c[11];
z q[1];
z q[4];
u1(1.5707963267948966) q[9];
tdg q[5];
u1(1.5707963267948966) q[0];
tdg q[5];
id q[5];
rz(1.5707963267948966) q[10];
sdg q[7];
s q[4];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[6];
z q[6];
sdg q[9];
id q[7];
id q[0];
sdg q[7];
z q[6];
s q[0];
s q[4];
id q[6];
t q[2];
s q[8];
s q[10];
id q[6];
t q[4];
z q[7];
u1(1.5707963267948966) q[0];
tdg q[4];
sdg q[5];
sdg q[8];
u1(1.5707963267948966) q[1];
t q[6];
z q[9];
id q[4];
sdg q[1];
tdg q[9];
s q[6];
z q[10];
tdg q[6];
tdg q[10];
sdg q[9];
id q[0];
s q[2];
z q[7];
z q[5];
t q[2];
u1(1.5707963267948966) q[6];
rz(1.5707963267948966) q[5];
t q[10];
sdg q[1];
tdg q[9];
t q[6];
t q[6];
u1(1.5707963267948966) q[9];
tdg q[9];
z q[0];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[6];
tdg q[0];
id q[4];
sdg q[9];
rz(1.5707963267948966) q[7];
tdg q[0];
sdg q[10];
z q[8];
rz(1.5707963267948966) q[5];
s q[9];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[6];
s q[0];
z q[7];
t q[8];
z q[7];
t q[4];
rz(1.5707963267948966) q[5];