OPENQASM 2.0;
include "qelib1.inc";
qreg q[29];
creg c[29];
sdg q[10];
tdg q[9];
s q[19];
sdg q[24];
sdg q[14];
id q[5];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[15];
rz(1.5707963267948966) q[27];
u1(1.5707963267948966) q[19];
id q[14];
rz(1.5707963267948966) q[23];
z q[10];
z q[18];
z q[6];
t q[25];
z q[6];
s q[20];
u1(1.5707963267948966) q[2];
id q[23];
tdg q[22];
sdg q[1];
t q[11];
sdg q[14];
id q[5];
z q[20];
sdg q[21];
tdg q[12];
u1(1.5707963267948966) q[9];
z q[24];
rz(1.5707963267948966) q[10];
z q[21];
z q[1];
u1(1.5707963267948966) q[24];
u1(1.5707963267948966) q[25];
u1(1.5707963267948966) q[17];
s q[8];
z q[8];
id q[5];
rz(1.5707963267948966) q[17];
id q[27];
t q[28];
t q[11];
z q[5];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[13];
sdg q[9];
id q[6];
sdg q[11];
tdg q[7];
sdg q[5];
z q[26];
u1(1.5707963267948966) q[13];
s q[19];
u1(1.5707963267948966) q[18];
id q[25];
tdg q[6];
tdg q[17];
z q[5];
s q[13];
s q[27];
s q[4];
id q[16];
s q[16];
id q[8];
rz(1.5707963267948966) q[15];
t q[26];
tdg q[25];
z q[28];
s q[12];
sdg q[23];
z q[10];
u1(1.5707963267948966) q[10];
sdg q[6];
rz(1.5707963267948966) q[15];
t q[18];
u1(1.5707963267948966) q[10];
u1(1.5707963267948966) q[21];
sdg q[23];
tdg q[15];
tdg q[15];
t q[0];
t q[8];
z q[14];
t q[2];
sdg q[26];
sdg q[26];
z q[9];
rz(1.5707963267948966) q[5];
s q[8];
z q[11];
s q[7];
id q[3];
id q[3];
id q[1];
id q[13];
tdg q[17];
s q[24];
rz(1.5707963267948966) q[18];
sdg q[27];
id q[8];
s q[28];
tdg q[24];
tdg q[23];
z q[0];
id q[11];
u1(1.5707963267948966) q[14];
rz(1.5707963267948966) q[3];
s q[21];
rz(1.5707963267948966) q[17];
tdg q[8];
id q[4];
z q[8];
sdg q[13];
id q[21];
tdg q[28];
t q[5];
t q[11];
s q[11];
z q[4];
id q[11];
rz(1.5707963267948966) q[9];
u1(1.5707963267948966) q[14];
s q[24];
z q[6];
z q[20];
s q[6];
rz(1.5707963267948966) q[17];
id q[17];
rz(1.5707963267948966) q[8];
t q[14];
sdg q[8];
id q[16];
rz(1.5707963267948966) q[3];
t q[12];
sdg q[17];
sdg q[5];
id q[9];
z q[25];
u1(1.5707963267948966) q[3];
s q[19];
s q[1];
rz(1.5707963267948966) q[18];
id q[27];
t q[28];
z q[20];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[7];
z q[11];
t q[22];
z q[6];
u1(1.5707963267948966) q[16];
tdg q[19];
z q[8];
id q[21];
rz(1.5707963267948966) q[9];
tdg q[21];
rz(1.5707963267948966) q[22];
z q[17];
z q[25];
sdg q[23];
u1(1.5707963267948966) q[25];
t q[25];
s q[7];
sdg q[3];
u1(1.5707963267948966) q[13];
u1(1.5707963267948966) q[14];
id q[2];
rz(1.5707963267948966) q[15];
tdg q[1];
rz(1.5707963267948966) q[23];
s q[1];
rz(1.5707963267948966) q[16];
u1(1.5707963267948966) q[24];
u1(1.5707963267948966) q[14];
tdg q[24];
t q[23];
id q[19];
s q[6];
s q[15];
tdg q[13];
rz(1.5707963267948966) q[4];
sdg q[2];
s q[5];
u1(1.5707963267948966) q[10];
z q[17];
t q[4];
tdg q[27];
id q[14];
id q[8];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[25];
tdg q[4];
t q[21];
z q[3];
rz(1.5707963267948966) q[13];
s q[1];
s q[25];
sdg q[6];
rz(1.5707963267948966) q[3];
id q[27];
s q[10];
z q[14];
u1(1.5707963267948966) q[22];
t q[20];
tdg q[8];
sdg q[4];
sdg q[17];
z q[23];
sdg q[26];
rz(1.5707963267948966) q[19];
t q[28];
s q[9];
u1(1.5707963267948966) q[6];
u1(1.5707963267948966) q[16];
u1(1.5707963267948966) q[23];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[10];
s q[16];
u1(1.5707963267948966) q[13];
t q[9];
t q[16];
t q[27];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[22];
id q[4];
rz(1.5707963267948966) q[19];
s q[8];
id q[16];
z q[27];
