OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
sdg q[7];
z q[10];
sdg q[8];
u1(1.5707963267948966) q[4];
sdg q[14];
id q[14];
rz(1.5707963267948966) q[7];
tdg q[14];
z q[15];
s q[15];
id q[8];
sdg q[17];
z q[5];
t q[17];
s q[6];
z q[18];
sdg q[9];
sdg q[18];
t q[12];
z q[18];
sdg q[12];
z q[7];
t q[13];
t q[4];
rz(1.5707963267948966) q[10];
id q[9];
rz(1.5707963267948966) q[9];
s q[1];
z q[4];
id q[10];
u1(1.5707963267948966) q[3];
t q[2];
rz(1.5707963267948966) q[18];
id q[16];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[4];
id q[6];
rz(1.5707963267948966) q[2];
z q[7];
sdg q[5];
z q[2];
u1(1.5707963267948966) q[9];
z q[14];
tdg q[13];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[12];
s q[18];
s q[1];
tdg q[2];
rz(1.5707963267948966) q[6];
tdg q[12];
t q[11];
rz(1.5707963267948966) q[12];
s q[2];
id q[13];
z q[10];
tdg q[5];
sdg q[2];
s q[2];
sdg q[13];
u1(1.5707963267948966) q[12];
tdg q[9];
tdg q[13];
rz(1.5707963267948966) q[11];
s q[3];
sdg q[13];
t q[6];
tdg q[17];
id q[8];
z q[6];
u1(1.5707963267948966) q[3];
id q[14];
sdg q[7];
u1(1.5707963267948966) q[0];
s q[5];
z q[1];
rz(1.5707963267948966) q[16];
sdg q[16];
z q[7];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[18];
id q[15];
s q[13];
sdg q[10];
rz(1.5707963267948966) q[8];
t q[4];
tdg q[0];
z q[9];
sdg q[8];
tdg q[1];
tdg q[3];
tdg q[3];
rz(1.5707963267948966) q[4];
sdg q[14];
rz(1.5707963267948966) q[4];
t q[5];
s q[2];
id q[14];
rz(1.5707963267948966) q[14];
u1(1.5707963267948966) q[7];
z q[11];
z q[16];
sdg q[18];
u1(1.5707963267948966) q[5];
sdg q[8];
z q[1];
u1(1.5707963267948966) q[16];
sdg q[13];
sdg q[15];
tdg q[5];
u1(1.5707963267948966) q[10];
t q[4];
rz(1.5707963267948966) q[16];
id q[17];
z q[17];
rz(1.5707963267948966) q[12];
s q[13];
t q[4];
sdg q[3];
id q[3];
sdg q[12];
s q[11];
rz(1.5707963267948966) q[8];
id q[6];
sdg q[15];
id q[0];
id q[11];
u1(1.5707963267948966) q[0];
t q[12];
s q[11];
s q[12];
u1(1.5707963267948966) q[5];
t q[3];
rz(1.5707963267948966) q[12];
tdg q[13];
rz(1.5707963267948966) q[12];
t q[1];
t q[2];
rz(1.5707963267948966) q[12];
sdg q[2];
s q[15];
u1(1.5707963267948966) q[0];
z q[13];
t q[5];
s q[3];
s q[2];
s q[4];
tdg q[4];
t q[3];
rz(1.5707963267948966) q[11];
t q[12];
u1(1.5707963267948966) q[5];
