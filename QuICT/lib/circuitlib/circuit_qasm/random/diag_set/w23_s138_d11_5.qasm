OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[21];
tdg q[20];
u1(1.5707963267948966) q[21];
z q[12];
t q[18];
id q[13];
s q[21];
rz(1.5707963267948966) q[0];
s q[19];
tdg q[1];
rz(1.5707963267948966) q[11];
tdg q[1];
sdg q[4];
tdg q[3];
tdg q[1];
rz(1.5707963267948966) q[14];
id q[22];
id q[7];
id q[16];
tdg q[21];
tdg q[21];
tdg q[18];
tdg q[3];
tdg q[11];
tdg q[3];
s q[1];
u1(1.5707963267948966) q[12];
id q[7];
id q[22];
id q[12];
sdg q[3];
sdg q[13];
s q[2];
rz(1.5707963267948966) q[21];
s q[19];
s q[8];
z q[0];
t q[4];
z q[20];
tdg q[17];
z q[5];
u1(1.5707963267948966) q[3];
t q[16];
tdg q[2];
rz(1.5707963267948966) q[17];
id q[19];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[14];
s q[17];
u1(1.5707963267948966) q[11];
id q[1];
t q[20];
tdg q[4];
sdg q[17];
s q[20];
s q[6];
id q[22];
id q[4];
s q[2];
rz(1.5707963267948966) q[6];
u1(1.5707963267948966) q[14];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[0];
s q[18];
t q[19];
t q[19];
id q[18];
tdg q[17];
id q[7];
s q[20];
tdg q[16];
z q[20];
t q[8];
tdg q[8];
rz(1.5707963267948966) q[3];
tdg q[1];
s q[5];
tdg q[6];
u1(1.5707963267948966) q[10];
tdg q[9];
tdg q[14];
id q[19];
id q[16];
rz(1.5707963267948966) q[21];
sdg q[0];
z q[2];
z q[15];
z q[13];
t q[7];
id q[16];
sdg q[22];
u1(1.5707963267948966) q[3];
z q[8];
z q[6];
tdg q[11];
id q[22];
tdg q[0];
s q[1];
t q[15];
rz(1.5707963267948966) q[17];
tdg q[18];
u1(1.5707963267948966) q[3];
sdg q[20];
s q[4];
rz(1.5707963267948966) q[17];
s q[16];
sdg q[4];
id q[16];
t q[3];
tdg q[3];
sdg q[14];
sdg q[11];
id q[7];
rz(1.5707963267948966) q[18];
t q[6];
sdg q[19];
tdg q[20];
s q[14];
t q[6];
tdg q[0];
t q[20];
sdg q[15];
u1(1.5707963267948966) q[11];
rz(1.5707963267948966) q[16];
z q[3];
id q[13];
t q[11];
z q[6];
s q[8];
tdg q[2];
sdg q[10];
u1(1.5707963267948966) q[21];
rz(1.5707963267948966) q[22];
s q[6];
sdg q[18];
t q[4];
id q[1];
