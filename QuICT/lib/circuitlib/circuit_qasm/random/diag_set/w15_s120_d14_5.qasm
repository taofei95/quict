OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[10];
u1(1.5707963267948966) q[14];
sdg q[9];
s q[3];
sdg q[3];
rz(1.5707963267948966) q[2];
sdg q[14];
id q[13];
z q[13];
z q[3];
t q[11];
t q[1];
s q[1];
s q[8];
id q[0];
u1(1.5707963267948966) q[11];
sdg q[14];
z q[6];
t q[2];
sdg q[3];
z q[14];
s q[1];
tdg q[0];
z q[3];
z q[9];
z q[14];
sdg q[14];
sdg q[1];
u1(1.5707963267948966) q[4];
z q[12];
rz(1.5707963267948966) q[1];
s q[4];
s q[12];
rz(1.5707963267948966) q[9];
t q[12];
id q[6];
z q[2];
z q[5];
id q[13];
z q[3];
z q[12];
z q[10];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
sdg q[11];
s q[14];
id q[9];
id q[9];
z q[4];
t q[10];
z q[4];
z q[14];
t q[10];
u1(1.5707963267948966) q[8];
sdg q[2];
tdg q[14];
t q[4];
id q[13];
z q[0];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[14];
s q[0];
z q[5];
sdg q[10];
s q[9];
t q[14];
u1(1.5707963267948966) q[8];
s q[12];
t q[9];
s q[9];
t q[10];
z q[12];
rz(1.5707963267948966) q[13];
id q[12];
s q[10];
rz(1.5707963267948966) q[14];
tdg q[7];
id q[11];
t q[6];
t q[2];
s q[13];
s q[0];
rz(1.5707963267948966) q[7];
sdg q[8];
sdg q[12];
z q[0];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[12];
s q[9];
id q[0];
z q[2];
t q[14];
tdg q[4];
tdg q[2];
sdg q[8];
sdg q[9];
tdg q[1];
id q[1];
t q[4];
rz(1.5707963267948966) q[4];
s q[8];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[10];
s q[12];
rz(1.5707963267948966) q[1];
s q[9];
id q[5];
id q[11];
s q[13];
z q[2];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[6];
t q[5];
rz(1.5707963267948966) q[4];
z q[1];
tdg q[10];
rz(1.5707963267948966) q[8];
id q[4];
