OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
t q[16];
t q[11];
sdg q[2];
t q[19];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[11];
u1(1.5707963267948966) q[19];
u1(1.5707963267948966) q[9];
u1(1.5707963267948966) q[0];
id q[0];
t q[1];
sdg q[6];
id q[5];
t q[6];
u1(1.5707963267948966) q[0];
t q[2];
sdg q[18];
t q[14];
t q[2];
s q[16];
s q[6];
sdg q[17];
rz(1.5707963267948966) q[6];
sdg q[2];
id q[7];
u1(1.5707963267948966) q[10];
id q[10];
tdg q[17];
id q[18];
sdg q[16];
t q[0];
tdg q[8];
tdg q[0];
tdg q[11];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[11];
rz(1.5707963267948966) q[0];
z q[5];
s q[6];
z q[5];
sdg q[17];
sdg q[3];
id q[10];
z q[18];
s q[8];
z q[9];
t q[3];
u1(1.5707963267948966) q[8];
id q[0];
t q[3];
sdg q[5];
tdg q[3];
id q[1];
sdg q[3];
s q[8];
u1(1.5707963267948966) q[12];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[13];
z q[16];
z q[19];
t q[15];
z q[10];
sdg q[14];
u1(1.5707963267948966) q[1];
tdg q[12];
u1(1.5707963267948966) q[4];
sdg q[19];
u1(1.5707963267948966) q[13];
u1(1.5707963267948966) q[10];
tdg q[8];
tdg q[15];
id q[10];
t q[18];
u1(1.5707963267948966) q[11];
id q[13];
sdg q[18];
sdg q[15];
z q[10];
z q[13];
z q[11];
z q[12];
z q[16];
sdg q[5];
s q[3];
t q[2];
id q[8];
tdg q[14];
s q[13];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[13];
sdg q[5];
id q[17];
sdg q[6];
u1(1.5707963267948966) q[14];
s q[19];
rz(1.5707963267948966) q[13];
tdg q[5];
z q[5];
t q[2];
z q[16];
t q[12];
tdg q[3];
id q[0];
u1(1.5707963267948966) q[10];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
tdg q[7];
z q[2];
u1(1.5707963267948966) q[0];
t q[14];
sdg q[8];
u1(1.5707963267948966) q[3];
tdg q[2];
s q[19];
z q[14];
u1(1.5707963267948966) q[12];
