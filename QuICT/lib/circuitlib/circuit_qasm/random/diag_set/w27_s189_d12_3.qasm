OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[27];
sdg q[15];
t q[24];
tdg q[24];
z q[14];
u1(1.5707963267948966) q[9];
u1(1.5707963267948966) q[12];
id q[10];
z q[14];
sdg q[6];
sdg q[15];
rz(1.5707963267948966) q[8];
tdg q[21];
tdg q[23];
t q[23];
sdg q[20];
tdg q[17];
tdg q[23];
id q[7];
sdg q[2];
s q[6];
sdg q[24];
sdg q[14];
id q[13];
z q[11];
tdg q[10];
tdg q[7];
tdg q[4];
id q[11];
u1(1.5707963267948966) q[4];
t q[7];
s q[13];
s q[13];
u1(1.5707963267948966) q[24];
t q[13];
rz(1.5707963267948966) q[16];
id q[12];
t q[0];
rz(1.5707963267948966) q[15];
z q[18];
u1(1.5707963267948966) q[2];
z q[18];
tdg q[9];
z q[24];
id q[8];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
s q[24];
tdg q[17];
rz(1.5707963267948966) q[26];
id q[18];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[17];
s q[25];
rz(1.5707963267948966) q[26];
s q[23];
z q[1];
u1(1.5707963267948966) q[10];
z q[26];
id q[12];
u1(1.5707963267948966) q[16];
tdg q[16];
z q[25];
tdg q[11];
z q[17];
id q[7];
rz(1.5707963267948966) q[7];
tdg q[15];
t q[7];
tdg q[4];
tdg q[20];
sdg q[17];
u1(1.5707963267948966) q[26];
s q[18];
t q[24];
z q[6];
tdg q[8];
rz(1.5707963267948966) q[23];
t q[17];
z q[7];
tdg q[16];
id q[11];
s q[8];
s q[8];
t q[0];
t q[4];
z q[13];
rz(1.5707963267948966) q[14];
z q[1];
z q[24];
sdg q[3];
id q[4];
z q[8];
s q[21];
tdg q[17];
id q[9];
u1(1.5707963267948966) q[8];
sdg q[26];
s q[4];
tdg q[19];
sdg q[15];
t q[23];
tdg q[3];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[6];
u1(1.5707963267948966) q[7];
t q[3];
t q[9];
z q[11];
t q[20];
rz(1.5707963267948966) q[25];
id q[21];
s q[21];
id q[8];
z q[4];
t q[9];
t q[17];
sdg q[17];
rz(1.5707963267948966) q[21];
z q[18];
s q[25];
id q[12];
u1(1.5707963267948966) q[16];
rz(1.5707963267948966) q[6];
tdg q[19];
tdg q[22];
t q[14];
tdg q[2];
z q[2];
s q[4];
sdg q[9];
sdg q[23];
t q[7];
rz(1.5707963267948966) q[25];
u1(1.5707963267948966) q[0];
sdg q[26];
s q[11];
s q[19];
tdg q[13];
rz(1.5707963267948966) q[10];
z q[24];
t q[26];
u1(1.5707963267948966) q[18];
rz(1.5707963267948966) q[0];
tdg q[26];
u1(1.5707963267948966) q[21];
t q[10];
tdg q[23];
z q[15];
tdg q[8];
tdg q[13];
t q[0];
u1(1.5707963267948966) q[12];
id q[6];
sdg q[2];
rz(1.5707963267948966) q[2];
z q[13];
s q[19];
sdg q[10];
s q[16];
sdg q[13];
id q[11];
id q[14];
z q[25];
sdg q[26];
tdg q[16];
id q[21];
s q[0];
sdg q[5];
id q[5];
s q[9];
sdg q[2];
t q[11];
z q[8];
tdg q[21];
s q[5];
sdg q[21];
rz(1.5707963267948966) q[2];
z q[8];
tdg q[15];
t q[1];
rz(1.5707963267948966) q[10];
u1(1.5707963267948966) q[12];
id q[19];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[22];
sdg q[18];
tdg q[17];
tdg q[8];
