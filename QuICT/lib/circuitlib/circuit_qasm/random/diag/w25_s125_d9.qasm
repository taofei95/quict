OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
sdg q[20];
s q[10];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[14];
s q[4];
id q[8];
z q[14];
rz(1.5707963267948966) q[3];
tdg q[17];
u1(1.5707963267948966) q[15];
s q[17];
u1(1.5707963267948966) q[12];
u1(1.5707963267948966) q[15];
t q[22];
u1(1.5707963267948966) q[18];
id q[4];
tdg q[4];
z q[23];
t q[18];
rz(1.5707963267948966) q[10];
id q[4];
id q[12];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[18];
u1(1.5707963267948966) q[4];
s q[2];
u1(1.5707963267948966) q[5];
t q[2];
u1(1.5707963267948966) q[14];
s q[4];
t q[23];
rz(1.5707963267948966) q[21];
z q[15];
id q[9];
rz(1.5707963267948966) q[21];
u1(1.5707963267948966) q[6];
s q[10];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[9];
sdg q[11];
u1(1.5707963267948966) q[4];
tdg q[14];
z q[15];
tdg q[3];
s q[3];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[0];
s q[3];
z q[10];
t q[8];
z q[18];
z q[5];
u1(1.5707963267948966) q[19];
rz(1.5707963267948966) q[20];
u1(1.5707963267948966) q[11];
tdg q[11];
z q[8];
tdg q[21];
sdg q[12];
tdg q[22];
t q[14];
id q[19];
tdg q[12];
u1(1.5707963267948966) q[12];
rz(1.5707963267948966) q[11];
t q[17];
z q[6];
rz(1.5707963267948966) q[23];
t q[15];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[10];
rz(1.5707963267948966) q[15];
u1(1.5707963267948966) q[21];
id q[14];
sdg q[15];
id q[20];
rz(1.5707963267948966) q[10];
t q[18];
t q[6];
id q[22];
t q[5];
u1(1.5707963267948966) q[15];
sdg q[5];
id q[17];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[20];
sdg q[20];
u1(1.5707963267948966) q[11];
s q[16];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[24];
u1(1.5707963267948966) q[14];
sdg q[9];
id q[13];
z q[23];
id q[11];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[22];
u1(1.5707963267948966) q[2];
tdg q[9];
z q[18];
id q[6];
id q[21];
sdg q[4];
z q[5];
s q[11];
u1(1.5707963267948966) q[22];
id q[22];
u1(1.5707963267948966) q[6];
s q[19];
t q[6];
sdg q[17];
t q[22];
t q[4];
rz(1.5707963267948966) q[18];
s q[2];
sdg q[13];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[20];
s q[20];
tdg q[14];
t q[5];
id q[5];
z q[22];
