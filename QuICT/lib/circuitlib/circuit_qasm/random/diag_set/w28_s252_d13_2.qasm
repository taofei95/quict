OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c[28];
sdg q[26];
tdg q[4];
tdg q[2];
s q[3];
tdg q[3];
t q[12];
id q[13];
t q[11];
t q[14];
tdg q[1];
z q[6];
z q[1];
tdg q[1];
s q[22];
sdg q[15];
s q[17];
t q[18];
id q[24];
id q[15];
u1(1.5707963267948966) q[3];
id q[16];
s q[11];
u1(1.5707963267948966) q[12];
z q[21];
z q[8];
tdg q[19];
t q[25];
sdg q[26];
id q[4];
t q[19];
z q[0];
sdg q[23];
s q[13];
id q[18];
z q[3];
id q[4];
t q[22];
tdg q[17];
z q[18];
t q[6];
t q[20];
rz(1.5707963267948966) q[11];
id q[21];
tdg q[15];
rz(1.5707963267948966) q[22];
tdg q[17];
z q[21];
sdg q[4];
z q[6];
id q[15];
u1(1.5707963267948966) q[18];
z q[17];
z q[24];
u1(1.5707963267948966) q[16];
rz(1.5707963267948966) q[22];
s q[21];
rz(1.5707963267948966) q[16];
t q[26];
id q[24];
t q[27];
s q[16];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[9];
t q[19];
u1(1.5707963267948966) q[8];
z q[4];
z q[10];
tdg q[10];
tdg q[23];
t q[0];
s q[16];
z q[2];
t q[16];
z q[13];
sdg q[4];
u1(1.5707963267948966) q[20];
t q[21];
u1(1.5707963267948966) q[20];
id q[17];
z q[3];
sdg q[23];
s q[22];
z q[9];
sdg q[24];
u1(1.5707963267948966) q[6];
u1(1.5707963267948966) q[25];
s q[9];
s q[1];
tdg q[3];
rz(1.5707963267948966) q[17];
z q[19];
z q[19];
t q[22];
tdg q[15];
rz(1.5707963267948966) q[7];
sdg q[27];
s q[27];
z q[4];
id q[9];
rz(1.5707963267948966) q[0];
sdg q[9];
u1(1.5707963267948966) q[26];
rz(1.5707963267948966) q[14];
tdg q[11];
s q[5];
t q[15];
t q[25];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[24];
tdg q[5];
s q[21];
t q[14];
sdg q[15];
s q[17];
tdg q[19];
rz(1.5707963267948966) q[9];
id q[17];
tdg q[25];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[22];
id q[1];
s q[13];
id q[6];
id q[8];
rz(1.5707963267948966) q[5];
s q[25];
id q[26];
tdg q[11];
id q[16];
sdg q[12];
t q[15];
id q[10];
tdg q[0];
rz(1.5707963267948966) q[26];
t q[17];
s q[18];
z q[23];
sdg q[19];
id q[7];
t q[7];
sdg q[10];
s q[27];
tdg q[25];
u1(1.5707963267948966) q[3];
id q[20];
sdg q[22];
z q[1];
rz(1.5707963267948966) q[4];
tdg q[4];
z q[14];
tdg q[2];
id q[0];
z q[25];
id q[25];
tdg q[21];
id q[27];
t q[13];
t q[3];
tdg q[4];
t q[7];
tdg q[7];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[20];
id q[12];
id q[25];
t q[2];
sdg q[19];
sdg q[19];
u1(1.5707963267948966) q[16];
rz(1.5707963267948966) q[25];
z q[12];
id q[8];
sdg q[19];
id q[8];
u1(1.5707963267948966) q[22];
tdg q[1];
z q[10];
t q[21];
rz(1.5707963267948966) q[2];
t q[13];
z q[6];
sdg q[0];
t q[9];
tdg q[12];
t q[17];
sdg q[0];
id q[10];
t q[16];
t q[5];
t q[15];
t q[20];
id q[2];
s q[18];
z q[8];
id q[5];
z q[2];
id q[18];
t q[10];
u1(1.5707963267948966) q[22];
t q[3];
rz(1.5707963267948966) q[25];
id q[20];
rz(1.5707963267948966) q[27];
t q[5];
t q[14];
id q[2];
tdg q[5];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[8];
z q[7];
id q[9];
t q[21];
s q[6];
sdg q[27];
id q[8];
s q[21];
sdg q[12];
u1(1.5707963267948966) q[16];
z q[0];
sdg q[17];
t q[27];
sdg q[1];
sdg q[20];
sdg q[19];
u1(1.5707963267948966) q[14];
u1(1.5707963267948966) q[3];
s q[1];
u1(1.5707963267948966) q[15];
s q[23];
u1(1.5707963267948966) q[6];
t q[7];
s q[5];
id q[22];
s q[18];
z q[14];
rz(1.5707963267948966) q[20];
sdg q[21];
z q[17];
id q[5];
s q[21];
tdg q[25];
s q[17];
sdg q[21];
sdg q[20];
tdg q[5];
id q[12];
rz(1.5707963267948966) q[19];
sdg q[3];
tdg q[12];
t q[4];
z q[27];
tdg q[26];
