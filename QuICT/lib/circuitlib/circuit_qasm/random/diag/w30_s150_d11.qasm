OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
rz(1.5707963267948966) q[16];
id q[1];
s q[26];
z q[2];
rz(1.5707963267948966) q[29];
t q[1];
u1(1.5707963267948966) q[9];
sdg q[14];
sdg q[4];
s q[14];
s q[4];
u1(1.5707963267948966) q[6];
rz(1.5707963267948966) q[13];
sdg q[9];
t q[14];
u1(1.5707963267948966) q[1];
z q[1];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[11];
sdg q[23];
tdg q[21];
u1(1.5707963267948966) q[25];
id q[2];
u1(1.5707963267948966) q[16];
t q[2];
tdg q[5];
u1(1.5707963267948966) q[16];
s q[0];
u1(1.5707963267948966) q[11];
s q[27];
rz(1.5707963267948966) q[4];
id q[24];
rz(1.5707963267948966) q[11];
s q[10];
s q[7];
id q[15];
t q[18];
sdg q[0];
rz(1.5707963267948966) q[0];
z q[20];
s q[9];
s q[5];
id q[13];
id q[5];
rz(1.5707963267948966) q[24];
s q[11];
s q[21];
z q[20];
sdg q[3];
u1(1.5707963267948966) q[12];
s q[4];
t q[27];
id q[13];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
id q[3];
s q[7];
rz(1.5707963267948966) q[26];
sdg q[3];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[19];
s q[14];
u1(1.5707963267948966) q[26];
rz(1.5707963267948966) q[1];
z q[2];
id q[29];
u1(1.5707963267948966) q[22];
sdg q[16];
rz(1.5707963267948966) q[17];
t q[3];
t q[17];
s q[14];
u1(1.5707963267948966) q[11];
z q[4];
t q[24];
s q[11];
u1(1.5707963267948966) q[12];
rz(1.5707963267948966) q[6];
z q[7];
id q[22];
id q[9];
tdg q[22];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[20];
z q[29];
id q[14];
z q[24];
z q[28];
tdg q[10];
id q[24];
z q[23];
rz(1.5707963267948966) q[7];
t q[3];
id q[17];
t q[13];
tdg q[4];
tdg q[8];
sdg q[9];
s q[23];
s q[0];
u1(1.5707963267948966) q[21];
t q[16];
id q[6];
t q[28];
u1(1.5707963267948966) q[4];
id q[28];
z q[11];
sdg q[1];
z q[24];
u1(1.5707963267948966) q[28];
s q[1];
s q[1];
sdg q[11];
s q[14];
id q[2];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[9];
sdg q[20];
t q[17];
u1(1.5707963267948966) q[8];
sdg q[1];
s q[18];
s q[23];
rz(1.5707963267948966) q[0];
sdg q[27];
s q[1];
rz(1.5707963267948966) q[13];
z q[22];
tdg q[25];
id q[22];
s q[23];
t q[13];
tdg q[5];
s q[14];
s q[27];
sdg q[14];
sdg q[15];
t q[21];
t q[7];
t q[10];
id q[0];
sdg q[7];
tdg q[26];
t q[18];
t q[23];
id q[6];
z q[18];
