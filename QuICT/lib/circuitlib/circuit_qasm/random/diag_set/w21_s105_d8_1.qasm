OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
sdg q[20];
tdg q[9];
z q[11];
id q[18];
z q[18];
sdg q[16];
rz(1.5707963267948966) q[8];
id q[4];
sdg q[16];
z q[20];
t q[9];
u1(1.5707963267948966) q[11];
s q[14];
id q[18];
rz(1.5707963267948966) q[2];
s q[19];
z q[4];
t q[15];
s q[1];
id q[18];
sdg q[1];
sdg q[2];
z q[5];
tdg q[9];
t q[7];
sdg q[10];
tdg q[16];
sdg q[13];
sdg q[15];
z q[18];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[19];
rz(1.5707963267948966) q[11];
u1(1.5707963267948966) q[6];
t q[16];
t q[15];
t q[19];
sdg q[9];
u1(1.5707963267948966) q[9];
id q[17];
sdg q[18];
rz(1.5707963267948966) q[20];
id q[14];
t q[7];
t q[13];
z q[15];
id q[20];
sdg q[5];
u1(1.5707963267948966) q[17];
sdg q[18];
id q[3];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[16];
id q[4];
u1(1.5707963267948966) q[17];
tdg q[1];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[11];
t q[8];
sdg q[17];
u1(1.5707963267948966) q[8];
s q[5];
id q[7];
u1(1.5707963267948966) q[1];
t q[15];
t q[7];
s q[5];
tdg q[2];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[8];
t q[15];
s q[6];
t q[17];
s q[10];
tdg q[6];
s q[0];
sdg q[2];
sdg q[7];
u1(1.5707963267948966) q[10];
z q[15];
sdg q[10];
rz(1.5707963267948966) q[14];
sdg q[18];
z q[9];
t q[6];
t q[20];
id q[14];
z q[17];
tdg q[16];
z q[6];
sdg q[9];
tdg q[13];
z q[8];
s q[14];
z q[11];
sdg q[2];
z q[8];
tdg q[1];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[11];
t q[10];
sdg q[9];
