OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
z q[1];
tdg q[4];
t q[7];
u1(1.5707963267948966) q[13];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[15];
tdg q[5];
tdg q[15];
rz(1.5707963267948966) q[3];
z q[2];
id q[1];
u1(1.5707963267948966) q[15];
z q[13];
sdg q[23];
rz(1.5707963267948966) q[11];
id q[5];
rz(1.5707963267948966) q[7];
t q[22];
tdg q[10];
sdg q[10];
z q[6];
z q[20];
z q[9];
sdg q[23];
u1(1.5707963267948966) q[13];
u1(1.5707963267948966) q[24];
rz(1.5707963267948966) q[11];
tdg q[5];
sdg q[20];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[19];
sdg q[2];
sdg q[10];
s q[12];
tdg q[0];
u1(1.5707963267948966) q[24];
s q[19];
z q[3];
u1(1.5707963267948966) q[3];
z q[19];
t q[15];
tdg q[18];
tdg q[19];
u1(1.5707963267948966) q[8];
s q[9];
u1(1.5707963267948966) q[22];
id q[7];
tdg q[4];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[9];
s q[2];
tdg q[3];
id q[11];
t q[5];
u1(1.5707963267948966) q[11];
u1(1.5707963267948966) q[15];
u1(1.5707963267948966) q[8];
sdg q[18];
u1(1.5707963267948966) q[10];
t q[22];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[13];
tdg q[4];
rz(1.5707963267948966) q[22];
t q[15];
rz(1.5707963267948966) q[22];
id q[20];
s q[13];
z q[21];
s q[17];
u1(1.5707963267948966) q[23];
sdg q[8];
id q[23];
u1(1.5707963267948966) q[1];
z q[24];
sdg q[21];
id q[14];
sdg q[14];
id q[24];
z q[19];
z q[9];
u1(1.5707963267948966) q[11];
tdg q[23];
z q[0];
t q[22];
id q[3];
s q[8];
sdg q[15];
sdg q[12];
u1(1.5707963267948966) q[17];
id q[8];
t q[20];
sdg q[20];
z q[1];
tdg q[22];
s q[15];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[15];
sdg q[9];
sdg q[12];
t q[14];
rz(1.5707963267948966) q[8];
z q[21];
z q[0];
rz(1.5707963267948966) q[10];
s q[18];
id q[0];
sdg q[21];
tdg q[9];
z q[24];
t q[0];
z q[5];
sdg q[9];
s q[1];
id q[1];
z q[24];
rz(1.5707963267948966) q[4];
sdg q[6];
u1(1.5707963267948966) q[22];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[23];
z q[23];
u1(1.5707963267948966) q[18];
rz(1.5707963267948966) q[3];
s q[0];
rz(1.5707963267948966) q[22];
s q[5];
s q[23];
sdg q[5];
id q[19];
z q[16];
s q[11];
t q[14];
tdg q[9];
id q[0];
sdg q[14];
s q[8];
id q[6];
z q[13];
t q[12];
id q[23];
rz(1.5707963267948966) q[16];
sdg q[6];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[2];
id q[24];
u1(1.5707963267948966) q[19];
z q[20];
s q[7];
tdg q[10];
u1(1.5707963267948966) q[11];
rz(1.5707963267948966) q[8];
s q[6];
u1(1.5707963267948966) q[4];
tdg q[23];
tdg q[5];
u1(1.5707963267948966) q[2];
sdg q[8];
rz(1.5707963267948966) q[16];
tdg q[3];
tdg q[24];
z q[11];
z q[16];
t q[23];
id q[20];
z q[13];
sdg q[11];
u1(1.5707963267948966) q[8];
t q[2];
t q[11];
tdg q[21];
t q[2];
tdg q[4];
u1(1.5707963267948966) q[12];
tdg q[7];
