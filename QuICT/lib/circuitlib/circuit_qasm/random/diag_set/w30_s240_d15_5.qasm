OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[7];
t q[25];
t q[18];
sdg q[17];
z q[3];
u1(1.5707963267948966) q[2];
t q[20];
t q[9];
rz(1.5707963267948966) q[16];
t q[22];
tdg q[4];
rz(1.5707963267948966) q[15];
id q[16];
sdg q[13];
id q[2];
t q[25];
u1(1.5707963267948966) q[24];
u1(1.5707963267948966) q[11];
u1(1.5707963267948966) q[6];
u1(1.5707963267948966) q[21];
s q[15];
z q[20];
t q[14];
s q[25];
t q[4];
id q[21];
rz(1.5707963267948966) q[14];
t q[29];
sdg q[22];
tdg q[6];
t q[28];
rz(1.5707963267948966) q[1];
t q[23];
tdg q[9];
rz(1.5707963267948966) q[29];
t q[22];
s q[8];
u1(1.5707963267948966) q[11];
z q[20];
sdg q[24];
tdg q[0];
s q[8];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[22];
id q[12];
tdg q[13];
sdg q[0];
sdg q[18];
tdg q[6];
id q[24];
rz(1.5707963267948966) q[17];
sdg q[13];
tdg q[29];
z q[0];
rz(1.5707963267948966) q[18];
s q[1];
sdg q[29];
t q[8];
s q[4];
t q[18];
z q[13];
s q[15];
z q[2];
rz(1.5707963267948966) q[17];
u1(1.5707963267948966) q[6];
u1(1.5707963267948966) q[13];
rz(1.5707963267948966) q[28];
t q[27];
s q[5];
s q[29];
z q[1];
u1(1.5707963267948966) q[8];
z q[4];
z q[22];
id q[18];
sdg q[4];
sdg q[6];
u1(1.5707963267948966) q[8];
id q[26];
rz(1.5707963267948966) q[13];
z q[14];
rz(1.5707963267948966) q[24];
u1(1.5707963267948966) q[13];
sdg q[10];
t q[19];
u1(1.5707963267948966) q[2];
s q[22];
id q[26];
s q[11];
s q[6];
sdg q[21];
u1(1.5707963267948966) q[18];
tdg q[8];
u1(1.5707963267948966) q[24];
rz(1.5707963267948966) q[28];
id q[2];
z q[29];
u1(1.5707963267948966) q[9];
t q[25];
s q[25];
s q[9];
rz(1.5707963267948966) q[22];
u1(1.5707963267948966) q[11];
id q[2];
s q[18];
z q[1];
z q[5];
rz(1.5707963267948966) q[9];
t q[25];
u1(1.5707963267948966) q[15];
t q[14];
tdg q[8];
s q[19];
rz(1.5707963267948966) q[4];
tdg q[13];
rz(1.5707963267948966) q[4];
t q[23];
z q[13];
t q[22];
id q[17];
z q[20];
s q[16];
id q[9];
rz(1.5707963267948966) q[17];
id q[20];
s q[18];
id q[23];
t q[2];
tdg q[25];
t q[11];
s q[9];
z q[1];
s q[7];
z q[26];
z q[26];
s q[26];
z q[11];
t q[6];
id q[19];
t q[24];
t q[17];
z q[18];
rz(1.5707963267948966) q[13];
z q[26];
t q[13];
t q[20];
tdg q[26];
rz(1.5707963267948966) q[5];
tdg q[13];
id q[2];
s q[25];
sdg q[11];
rz(1.5707963267948966) q[5];
tdg q[4];
id q[17];
tdg q[25];
tdg q[24];
s q[18];
id q[7];
s q[6];
t q[15];
id q[13];
s q[11];
sdg q[17];
tdg q[22];
s q[6];
tdg q[19];
tdg q[12];
tdg q[12];
u1(1.5707963267948966) q[28];
s q[0];
rz(1.5707963267948966) q[4];
tdg q[10];
s q[21];
sdg q[2];
tdg q[4];
sdg q[25];
sdg q[13];
t q[6];
t q[8];
z q[23];
tdg q[22];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[11];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[23];
t q[4];
tdg q[0];
sdg q[11];
t q[6];
z q[12];
sdg q[17];
tdg q[26];
s q[1];
rz(1.5707963267948966) q[14];
tdg q[20];
id q[11];
tdg q[17];
id q[11];
u1(1.5707963267948966) q[14];
id q[25];
sdg q[1];
rz(1.5707963267948966) q[3];
s q[25];
rz(1.5707963267948966) q[21];
s q[27];
s q[5];
rz(1.5707963267948966) q[9];
s q[8];
u1(1.5707963267948966) q[6];
id q[12];
id q[18];
id q[16];
t q[22];
z q[2];
id q[0];
u1(1.5707963267948966) q[21];
sdg q[8];
s q[27];
tdg q[18];
s q[18];
s q[21];
s q[8];
z q[21];
sdg q[6];
tdg q[16];
u1(1.5707963267948966) q[26];
sdg q[7];
rz(1.5707963267948966) q[27];
sdg q[13];
tdg q[17];
id q[24];
t q[0];
s q[18];
rz(1.5707963267948966) q[23];
s q[29];
z q[22];
sdg q[18];
id q[8];
