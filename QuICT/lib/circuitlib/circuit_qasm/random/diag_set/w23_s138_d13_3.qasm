OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
id q[11];
id q[22];
z q[21];
tdg q[22];
z q[2];
t q[3];
rz(1.5707963267948966) q[12];
id q[20];
u1(1.5707963267948966) q[14];
rz(1.5707963267948966) q[20];
z q[3];
id q[15];
s q[2];
rz(1.5707963267948966) q[22];
s q[9];
z q[19];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[8];
z q[3];
s q[16];
s q[14];
z q[9];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[18];
id q[15];
z q[5];
id q[14];
z q[15];
rz(1.5707963267948966) q[22];
sdg q[10];
u1(1.5707963267948966) q[9];
tdg q[3];
sdg q[3];
tdg q[13];
rz(1.5707963267948966) q[19];
id q[2];
z q[16];
rz(1.5707963267948966) q[11];
z q[19];
u1(1.5707963267948966) q[21];
s q[2];
z q[14];
u1(1.5707963267948966) q[6];
rz(1.5707963267948966) q[15];
sdg q[12];
t q[19];
s q[10];
z q[18];
rz(1.5707963267948966) q[0];
tdg q[5];
z q[15];
tdg q[13];
t q[19];
tdg q[3];
u1(1.5707963267948966) q[9];
rz(1.5707963267948966) q[16];
z q[8];
t q[21];
id q[14];
t q[11];
t q[9];
u1(1.5707963267948966) q[0];
sdg q[8];
t q[15];
u1(1.5707963267948966) q[4];
id q[9];
s q[7];
id q[0];
id q[9];
tdg q[17];
tdg q[20];
id q[0];
s q[1];
rz(1.5707963267948966) q[3];
z q[15];
u1(1.5707963267948966) q[15];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[17];
tdg q[3];
id q[8];
id q[20];
z q[2];
u1(1.5707963267948966) q[6];
u1(1.5707963267948966) q[11];
t q[22];
u1(1.5707963267948966) q[10];
tdg q[4];
z q[6];
u1(1.5707963267948966) q[10];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[14];
z q[15];
tdg q[17];
t q[8];
id q[3];
tdg q[8];
sdg q[5];
sdg q[2];
tdg q[7];
s q[1];
z q[15];
sdg q[14];
u1(1.5707963267948966) q[11];
id q[18];
rz(1.5707963267948966) q[4];
z q[9];
s q[15];
u1(1.5707963267948966) q[9];
sdg q[8];
tdg q[2];
s q[8];
id q[21];
id q[20];
u1(1.5707963267948966) q[0];
z q[15];
u1(1.5707963267948966) q[12];
rz(1.5707963267948966) q[10];
z q[18];
rz(1.5707963267948966) q[8];
z q[22];
z q[17];
t q[10];
rz(1.5707963267948966) q[6];
id q[11];
id q[9];
id q[21];
s q[14];
u1(1.5707963267948966) q[9];
id q[21];
sdg q[5];
s q[1];
id q[16];
rz(1.5707963267948966) q[1];
s q[21];
sdg q[7];
z q[17];
tdg q[22];
id q[15];
