OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
z q[12];
s q[12];
sdg q[6];
z q[8];
id q[5];
sdg q[12];
rz(1.5707963267948966) q[1];
sdg q[0];
t q[1];
id q[12];
sdg q[5];
t q[5];
sdg q[1];
id q[8];
id q[0];
t q[2];
t q[10];
s q[5];
s q[12];
t q[7];
sdg q[0];
sdg q[6];
s q[4];
rz(1.5707963267948966) q[9];
t q[0];
sdg q[10];
z q[12];
z q[2];
s q[1];
t q[10];
s q[11];
u1(1.5707963267948966) q[9];
id q[9];
tdg q[7];
u1(1.5707963267948966) q[12];
sdg q[3];
sdg q[8];
tdg q[10];
t q[3];
u1(1.5707963267948966) q[8];
t q[4];
s q[5];
s q[6];
z q[0];
t q[5];
s q[11];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[3];
s q[7];
id q[0];
z q[1];
tdg q[11];
rz(1.5707963267948966) q[11];
tdg q[1];
rz(1.5707963267948966) q[5];
z q[4];
u1(1.5707963267948966) q[11];
s q[0];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[4];
tdg q[10];
t q[4];
t q[2];
t q[8];
z q[11];
sdg q[6];
s q[1];
u1(1.5707963267948966) q[0];
id q[5];
t q[8];
id q[0];
sdg q[12];
id q[1];
sdg q[7];
tdg q[9];
t q[10];
u1(1.5707963267948966) q[5];
s q[10];
