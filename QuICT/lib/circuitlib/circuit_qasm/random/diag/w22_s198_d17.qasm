OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
z q[6];
id q[19];
id q[20];
t q[0];
rz(1.5707963267948966) q[6];
id q[7];
u1(1.5707963267948966) q[6];
t q[10];
sdg q[19];
rz(1.5707963267948966) q[17];
sdg q[1];
tdg q[4];
t q[15];
z q[0];
rz(1.5707963267948966) q[14];
z q[11];
tdg q[13];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[0];
sdg q[0];
u1(1.5707963267948966) q[7];
sdg q[15];
id q[18];
id q[20];
id q[4];
tdg q[15];
sdg q[0];
tdg q[5];
u1(1.5707963267948966) q[14];
sdg q[4];
tdg q[6];
z q[7];
z q[8];
sdg q[8];
rz(1.5707963267948966) q[12];
z q[8];
t q[18];
rz(1.5707963267948966) q[7];
s q[4];
z q[9];
id q[0];
id q[7];
z q[0];
id q[5];
s q[17];
t q[9];
id q[0];
rz(1.5707963267948966) q[18];
t q[13];
id q[18];
sdg q[15];
id q[0];
t q[10];
u1(1.5707963267948966) q[6];
tdg q[19];
u1(1.5707963267948966) q[21];
u1(1.5707963267948966) q[19];
t q[21];
z q[4];
rz(1.5707963267948966) q[20];
sdg q[14];
s q[12];
z q[15];
z q[15];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[7];
z q[18];
t q[4];
rz(1.5707963267948966) q[13];
z q[11];
rz(1.5707963267948966) q[18];
z q[6];
t q[21];
rz(1.5707963267948966) q[11];
sdg q[4];
u1(1.5707963267948966) q[2];
z q[7];
t q[4];
t q[6];
tdg q[8];
rz(1.5707963267948966) q[14];
tdg q[17];
s q[0];
u1(1.5707963267948966) q[2];
z q[9];
tdg q[14];
s q[18];
t q[5];
t q[20];
sdg q[10];
t q[15];
z q[4];
t q[14];
s q[11];
rz(1.5707963267948966) q[20];
z q[7];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[11];
rz(1.5707963267948966) q[13];
z q[14];
t q[16];
t q[13];
u1(1.5707963267948966) q[18];
tdg q[0];
s q[18];
id q[21];
t q[1];
u1(1.5707963267948966) q[5];
z q[18];
t q[17];
id q[13];
u1(1.5707963267948966) q[11];
s q[1];
s q[5];
sdg q[15];
rz(1.5707963267948966) q[12];
t q[7];
u1(1.5707963267948966) q[15];
s q[10];
rz(1.5707963267948966) q[15];
tdg q[2];
s q[11];
rz(1.5707963267948966) q[13];
t q[10];
sdg q[0];
sdg q[16];
id q[2];
z q[2];
rz(1.5707963267948966) q[18];
id q[19];
t q[8];
u1(1.5707963267948966) q[0];
sdg q[9];
id q[20];
tdg q[9];
rz(1.5707963267948966) q[10];
tdg q[8];
t q[19];
u1(1.5707963267948966) q[13];
t q[21];
tdg q[18];
u1(1.5707963267948966) q[21];
z q[6];
id q[7];
rz(1.5707963267948966) q[15];
t q[18];
z q[7];
t q[17];
rz(1.5707963267948966) q[14];
tdg q[13];
z q[4];
s q[21];
z q[8];
s q[12];
id q[7];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[7];
tdg q[10];
tdg q[14];
s q[7];
u1(1.5707963267948966) q[14];
t q[7];
sdg q[8];
s q[10];
u1(1.5707963267948966) q[21];
s q[6];
tdg q[3];
rz(1.5707963267948966) q[3];
sdg q[18];
s q[21];
z q[0];
t q[19];
t q[8];
sdg q[1];
tdg q[19];
rz(1.5707963267948966) q[12];
u1(1.5707963267948966) q[17];
tdg q[3];
u1(1.5707963267948966) q[12];
tdg q[16];
s q[7];
z q[18];
sdg q[20];
tdg q[20];
sdg q[5];
t q[3];
t q[7];
u1(1.5707963267948966) q[16];
sdg q[17];
rz(1.5707963267948966) q[19];
u1(1.5707963267948966) q[8];
id q[8];
s q[2];
id q[21];
tdg q[16];
id q[21];
s q[13];
