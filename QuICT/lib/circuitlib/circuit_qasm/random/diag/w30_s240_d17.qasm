OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
u1(1.5707963267948966) q[14];
u1(1.5707963267948966) q[25];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[0];
sdg q[23];
t q[13];
u1(1.5707963267948966) q[19];
z q[0];
s q[23];
rz(1.5707963267948966) q[28];
sdg q[15];
s q[23];
t q[15];
rz(1.5707963267948966) q[26];
t q[17];
t q[11];
s q[13];
z q[2];
tdg q[4];
s q[23];
s q[12];
z q[29];
tdg q[22];
s q[16];
rz(1.5707963267948966) q[3];
sdg q[11];
id q[5];
z q[1];
sdg q[23];
z q[19];
s q[4];
t q[26];
t q[7];
z q[5];
id q[14];
s q[9];
u1(1.5707963267948966) q[14];
s q[28];
z q[22];
s q[26];
rz(1.5707963267948966) q[15];
u1(1.5707963267948966) q[22];
rz(1.5707963267948966) q[15];
sdg q[29];
sdg q[9];
rz(1.5707963267948966) q[6];
t q[19];
s q[29];
s q[19];
sdg q[4];
t q[6];
z q[8];
id q[11];
t q[10];
id q[20];
s q[5];
z q[17];
rz(1.5707963267948966) q[16];
z q[12];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[13];
z q[26];
z q[9];
tdg q[7];
sdg q[0];
z q[6];
sdg q[7];
t q[5];
u1(1.5707963267948966) q[4];
sdg q[10];
tdg q[18];
z q[11];
rz(1.5707963267948966) q[27];
s q[12];
s q[13];
id q[11];
rz(1.5707963267948966) q[11];
tdg q[19];
u1(1.5707963267948966) q[2];
tdg q[19];
z q[24];
s q[9];
tdg q[21];
s q[4];
z q[13];
u1(1.5707963267948966) q[16];
t q[20];
z q[5];
id q[9];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[17];
t q[19];
sdg q[8];
z q[6];
t q[12];
id q[13];
rz(1.5707963267948966) q[3];
tdg q[9];
s q[24];
tdg q[5];
sdg q[1];
rz(1.5707963267948966) q[18];
id q[27];
u1(1.5707963267948966) q[29];
sdg q[8];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[1];
sdg q[8];
id q[12];
s q[24];
z q[25];
tdg q[12];
z q[12];
s q[11];
tdg q[11];
z q[7];
t q[29];
s q[6];
sdg q[2];
tdg q[1];
t q[16];
sdg q[26];
z q[24];
id q[26];
t q[28];
s q[3];
z q[9];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[13];
z q[21];
rz(1.5707963267948966) q[3];
t q[1];
t q[14];
id q[2];
id q[19];
sdg q[28];
tdg q[28];
id q[24];
t q[2];
u1(1.5707963267948966) q[15];
s q[28];
t q[0];
sdg q[15];
s q[19];
u1(1.5707963267948966) q[15];
rz(1.5707963267948966) q[27];
s q[28];
z q[5];
sdg q[20];
s q[8];
z q[29];
id q[8];
tdg q[26];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[17];
id q[6];
t q[2];
s q[8];
tdg q[2];
sdg q[19];
sdg q[29];
z q[9];
t q[19];
s q[22];
t q[11];
t q[8];
u1(1.5707963267948966) q[23];
z q[29];
rz(1.5707963267948966) q[20];
s q[9];
z q[21];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[14];
id q[3];
rz(1.5707963267948966) q[24];
tdg q[12];
tdg q[15];
u1(1.5707963267948966) q[19];
id q[3];
s q[13];
t q[7];
t q[14];
t q[7];
u1(1.5707963267948966) q[16];
z q[11];
sdg q[28];
z q[15];
t q[19];
u1(1.5707963267948966) q[19];
rz(1.5707963267948966) q[22];
s q[22];
s q[29];
s q[4];
t q[6];
t q[3];
sdg q[10];
rz(1.5707963267948966) q[26];
id q[19];
sdg q[26];
rz(1.5707963267948966) q[27];
t q[4];
s q[2];
u1(1.5707963267948966) q[22];
sdg q[7];
sdg q[12];
z q[11];
s q[9];
t q[28];
t q[14];
sdg q[10];
t q[19];
id q[28];
tdg q[5];
s q[14];
s q[23];
id q[9];
tdg q[16];
id q[5];
z q[8];
rz(1.5707963267948966) q[4];
tdg q[20];
sdg q[17];
t q[2];
z q[21];
rz(1.5707963267948966) q[2];
id q[7];
z q[17];
t q[20];
t q[18];
id q[26];
t q[13];
id q[26];
z q[28];
s q[10];
id q[16];
rz(1.5707963267948966) q[9];
z q[28];
u1(1.5707963267948966) q[27];
z q[0];
