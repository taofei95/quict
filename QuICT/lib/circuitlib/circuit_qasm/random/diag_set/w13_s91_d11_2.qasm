OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
rz(1.5707963267948966) q[1];
sdg q[5];
s q[3];
id q[0];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[4];
s q[6];
id q[10];
u1(1.5707963267948966) q[9];
t q[3];
u1(1.5707963267948966) q[2];
sdg q[7];
tdg q[9];
u1(1.5707963267948966) q[3];
s q[7];
id q[4];
id q[8];
tdg q[10];
t q[9];
rz(1.5707963267948966) q[9];
s q[4];
s q[10];
u1(1.5707963267948966) q[8];
sdg q[4];
sdg q[10];
id q[8];
rz(1.5707963267948966) q[4];
z q[1];
sdg q[11];
t q[3];
s q[12];
sdg q[2];
tdg q[3];
sdg q[7];
id q[12];
u1(1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
tdg q[12];
tdg q[3];
sdg q[2];
u1(1.5707963267948966) q[2];
s q[3];
z q[9];
rz(1.5707963267948966) q[12];
u1(1.5707963267948966) q[5];
id q[3];
u1(1.5707963267948966) q[11];
id q[10];
tdg q[6];
s q[9];
id q[8];
t q[6];
sdg q[11];
s q[12];
tdg q[9];
z q[7];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[8];
t q[8];
id q[0];
rz(1.5707963267948966) q[12];
z q[6];
sdg q[8];
t q[8];
s q[4];
u1(1.5707963267948966) q[12];
t q[7];
tdg q[0];
sdg q[1];
z q[4];
t q[8];
id q[11];
u1(1.5707963267948966) q[11];
tdg q[2];
t q[9];
tdg q[2];
tdg q[3];
rz(1.5707963267948966) q[6];
t q[1];
id q[3];
u1(1.5707963267948966) q[4];
s q[11];
z q[0];
tdg q[3];
t q[1];
t q[11];
rz(1.5707963267948966) q[2];
s q[2];
id q[6];
id q[11];
