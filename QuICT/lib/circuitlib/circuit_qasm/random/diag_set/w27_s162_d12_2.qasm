OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[27];
id q[26];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[11];
t q[26];
u1(1.5707963267948966) q[13];
id q[16];
s q[1];
s q[3];
sdg q[14];
s q[8];
z q[18];
s q[0];
sdg q[5];
z q[24];
s q[26];
z q[14];
t q[6];
t q[16];
z q[14];
t q[13];
rz(1.5707963267948966) q[13];
u1(1.5707963267948966) q[11];
t q[22];
t q[3];
u1(1.5707963267948966) q[16];
tdg q[21];
z q[25];
rz(1.5707963267948966) q[25];
u1(1.5707963267948966) q[22];
rz(1.5707963267948966) q[9];
z q[20];
z q[0];
id q[21];
u1(1.5707963267948966) q[21];
rz(1.5707963267948966) q[16];
id q[3];
z q[6];
id q[10];
tdg q[5];
t q[1];
id q[1];
t q[26];
u1(1.5707963267948966) q[0];
sdg q[15];
z q[11];
z q[26];
z q[9];
rz(1.5707963267948966) q[12];
t q[1];
t q[18];
t q[25];
rz(1.5707963267948966) q[0];
id q[7];
z q[22];
t q[18];
rz(1.5707963267948966) q[23];
id q[7];
tdg q[0];
tdg q[6];
z q[14];
tdg q[23];
rz(1.5707963267948966) q[24];
id q[13];
s q[6];
tdg q[17];
u1(1.5707963267948966) q[20];
sdg q[3];
u1(1.5707963267948966) q[18];
t q[24];
id q[12];
sdg q[1];
tdg q[1];
u1(1.5707963267948966) q[18];
t q[12];
t q[15];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[24];
rz(1.5707963267948966) q[1];
t q[21];
z q[19];
z q[11];
rz(1.5707963267948966) q[6];
tdg q[26];
rz(1.5707963267948966) q[3];
s q[17];
s q[10];
z q[16];
id q[26];
s q[22];
u1(1.5707963267948966) q[19];
u1(1.5707963267948966) q[1];
sdg q[5];
sdg q[10];
s q[7];
z q[18];
z q[21];
sdg q[7];
z q[20];
u1(1.5707963267948966) q[25];
u1(1.5707963267948966) q[20];
rz(1.5707963267948966) q[11];
z q[18];
u1(1.5707963267948966) q[10];
tdg q[17];
rz(1.5707963267948966) q[14];
id q[7];
rz(1.5707963267948966) q[13];
t q[24];
rz(1.5707963267948966) q[18];
z q[14];
id q[21];
s q[19];
s q[0];
tdg q[21];
t q[2];
z q[4];
s q[17];
t q[10];
u1(1.5707963267948966) q[15];
sdg q[12];
t q[3];
z q[11];
id q[12];
s q[6];
z q[21];
z q[16];
t q[5];
z q[19];
t q[11];
tdg q[18];
tdg q[4];
s q[4];
z q[14];
rz(1.5707963267948966) q[19];
sdg q[18];
s q[25];
t q[20];
id q[19];
id q[2];
s q[22];
id q[10];
rz(1.5707963267948966) q[9];
z q[23];
id q[10];
s q[23];
s q[0];
t q[11];
u1(1.5707963267948966) q[18];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[3];
sdg q[20];
rz(1.5707963267948966) q[11];
z q[7];
t q[14];
rz(1.5707963267948966) q[6];
t q[12];
u1(1.5707963267948966) q[21];
u1(1.5707963267948966) q[18];
u1(1.5707963267948966) q[17];
t q[14];
u1(1.5707963267948966) q[20];
t q[14];
