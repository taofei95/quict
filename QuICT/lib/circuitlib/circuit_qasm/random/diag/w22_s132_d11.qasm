OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
id q[4];
sdg q[13];
t q[16];
t q[18];
tdg q[18];
u1(1.5707963267948966) q[21];
s q[21];
tdg q[20];
s q[20];
sdg q[18];
tdg q[8];
s q[5];
rz(1.5707963267948966) q[9];
t q[10];
t q[15];
z q[5];
s q[4];
u1(1.5707963267948966) q[16];
z q[14];
rz(1.5707963267948966) q[8];
sdg q[9];
t q[11];
u1(1.5707963267948966) q[17];
sdg q[13];
rz(1.5707963267948966) q[12];
tdg q[5];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[1];
id q[8];
s q[2];
u1(1.5707963267948966) q[15];
tdg q[18];
z q[7];
z q[15];
z q[8];
t q[19];
z q[0];
tdg q[10];
s q[13];
rz(1.5707963267948966) q[2];
t q[18];
z q[3];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[17];
id q[14];
tdg q[14];
id q[3];
z q[15];
sdg q[0];
u1(1.5707963267948966) q[18];
z q[0];
z q[17];
z q[21];
rz(1.5707963267948966) q[13];
z q[6];
sdg q[8];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[8];
t q[7];
sdg q[10];
u1(1.5707963267948966) q[2];
id q[0];
z q[16];
sdg q[9];
rz(1.5707963267948966) q[8];
id q[0];
tdg q[17];
z q[7];
t q[13];
tdg q[2];
rz(1.5707963267948966) q[15];
u1(1.5707963267948966) q[20];
s q[17];
id q[5];
u1(1.5707963267948966) q[19];
tdg q[8];
t q[8];
sdg q[12];
id q[19];
u1(1.5707963267948966) q[18];
sdg q[21];
sdg q[13];
tdg q[14];
tdg q[5];
id q[0];
z q[15];
tdg q[15];
u1(1.5707963267948966) q[11];
z q[3];
s q[0];
u1(1.5707963267948966) q[20];
sdg q[14];
sdg q[14];
sdg q[16];
id q[7];
z q[3];
id q[0];
id q[19];
u1(1.5707963267948966) q[2];
t q[17];
u1(1.5707963267948966) q[20];
id q[7];
tdg q[16];
rz(1.5707963267948966) q[18];
tdg q[18];
tdg q[12];
id q[10];
z q[16];
tdg q[5];
sdg q[14];
tdg q[19];
tdg q[17];
id q[15];
t q[12];
sdg q[8];
z q[12];
id q[3];
u1(1.5707963267948966) q[9];
t q[10];
sdg q[15];
rz(1.5707963267948966) q[18];
z q[12];
u1(1.5707963267948966) q[17];
t q[16];
z q[4];
z q[20];
z q[11];
s q[11];
z q[12];
sdg q[21];
s q[18];
