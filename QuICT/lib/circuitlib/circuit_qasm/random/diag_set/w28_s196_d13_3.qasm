OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c[28];
s q[21];
t q[15];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[15];
rz(1.5707963267948966) q[4];
sdg q[3];
z q[18];
tdg q[23];
rz(1.5707963267948966) q[0];
z q[5];
tdg q[4];
z q[7];
rz(1.5707963267948966) q[3];
sdg q[4];
z q[25];
id q[3];
sdg q[9];
u1(1.5707963267948966) q[5];
t q[2];
id q[8];
t q[6];
u1(1.5707963267948966) q[14];
s q[2];
u1(1.5707963267948966) q[9];
t q[12];
rz(1.5707963267948966) q[21];
sdg q[8];
tdg q[16];
rz(1.5707963267948966) q[16];
t q[20];
rz(1.5707963267948966) q[27];
rz(1.5707963267948966) q[17];
id q[14];
sdg q[22];
t q[1];
tdg q[20];
tdg q[14];
tdg q[11];
tdg q[17];
z q[13];
id q[10];
tdg q[9];
tdg q[27];
id q[8];
rz(1.5707963267948966) q[11];
id q[22];
sdg q[22];
t q[10];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[20];
rz(1.5707963267948966) q[27];
t q[1];
id q[18];
t q[13];
sdg q[19];
rz(1.5707963267948966) q[5];
tdg q[3];
id q[7];
rz(1.5707963267948966) q[7];
sdg q[22];
z q[23];
id q[17];
s q[19];
u1(1.5707963267948966) q[10];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[8];
u1(1.5707963267948966) q[2];
z q[0];
sdg q[15];
u1(1.5707963267948966) q[19];
id q[0];
id q[10];
z q[6];
id q[22];
t q[23];
rz(1.5707963267948966) q[12];
u1(1.5707963267948966) q[5];
id q[24];
id q[21];
t q[18];
rz(1.5707963267948966) q[8];
s q[5];
u1(1.5707963267948966) q[26];
s q[17];
tdg q[12];
rz(1.5707963267948966) q[21];
id q[8];
tdg q[8];
s q[12];
s q[5];
tdg q[21];
z q[20];
s q[4];
id q[22];
z q[6];
id q[11];
s q[18];
tdg q[1];
z q[11];
sdg q[27];
u1(1.5707963267948966) q[4];
sdg q[11];
tdg q[14];
t q[15];
sdg q[19];
t q[13];
t q[11];
sdg q[21];
sdg q[17];
rz(1.5707963267948966) q[25];
id q[14];
u1(1.5707963267948966) q[15];
id q[6];
z q[26];
t q[15];
s q[19];
rz(1.5707963267948966) q[23];
z q[17];
rz(1.5707963267948966) q[27];
s q[16];
tdg q[8];
s q[14];
sdg q[23];
s q[25];
tdg q[12];
sdg q[21];
id q[23];
id q[26];
t q[20];
u1(1.5707963267948966) q[13];
id q[7];
u1(1.5707963267948966) q[6];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[0];
t q[15];
s q[9];
sdg q[23];
sdg q[24];
rz(1.5707963267948966) q[12];
u1(1.5707963267948966) q[27];
tdg q[20];
tdg q[13];
rz(1.5707963267948966) q[16];
id q[2];
tdg q[20];
t q[25];
z q[21];
rz(1.5707963267948966) q[22];
u1(1.5707963267948966) q[18];
tdg q[23];
t q[18];
tdg q[9];
z q[3];
id q[2];
s q[21];
t q[20];
t q[23];
u1(1.5707963267948966) q[8];
tdg q[19];
id q[23];
u1(1.5707963267948966) q[26];
tdg q[7];
t q[16];
t q[21];
s q[7];
id q[3];
z q[17];
sdg q[17];
tdg q[5];
u1(1.5707963267948966) q[13];
t q[7];
sdg q[5];
z q[27];
sdg q[8];
t q[23];
s q[6];
sdg q[19];
z q[23];
tdg q[5];
tdg q[1];
u1(1.5707963267948966) q[17];
s q[14];
t q[19];
id q[25];
id q[13];
t q[15];
sdg q[27];
z q[6];
t q[23];
rz(1.5707963267948966) q[15];
z q[4];
id q[1];
z q[25];
id q[21];
u1(1.5707963267948966) q[17];
u1(1.5707963267948966) q[19];
