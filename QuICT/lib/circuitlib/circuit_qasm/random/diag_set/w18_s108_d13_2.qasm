OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
u1(1.5707963267948966) q[10];
rz(1.5707963267948966) q[0];
z q[17];
id q[10];
s q[8];
u1(1.5707963267948966) q[10];
t q[5];
id q[12];
t q[9];
sdg q[2];
sdg q[12];
rz(1.5707963267948966) q[10];
t q[17];
s q[10];
rz(1.5707963267948966) q[0];
id q[14];
t q[4];
rz(1.5707963267948966) q[17];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[2];
z q[4];
u1(1.5707963267948966) q[16];
z q[8];
sdg q[1];
u1(1.5707963267948966) q[3];
t q[17];
t q[11];
z q[10];
rz(1.5707963267948966) q[12];
u1(1.5707963267948966) q[0];
s q[4];
t q[16];
id q[3];
id q[10];
tdg q[14];
sdg q[12];
t q[16];
tdg q[4];
sdg q[14];
z q[7];
s q[8];
rz(1.5707963267948966) q[13];
sdg q[8];
rz(1.5707963267948966) q[14];
tdg q[9];
u1(1.5707963267948966) q[2];
s q[12];
id q[10];
sdg q[15];
z q[4];
z q[12];
sdg q[16];
id q[5];
t q[0];
rz(1.5707963267948966) q[11];
t q[3];
z q[7];
s q[3];
z q[12];
sdg q[0];
rz(1.5707963267948966) q[11];
tdg q[5];
sdg q[1];
s q[0];
s q[15];
sdg q[12];
s q[2];
u1(1.5707963267948966) q[3];
z q[3];
t q[0];
u1(1.5707963267948966) q[15];
u1(1.5707963267948966) q[15];
s q[1];
s q[7];
s q[5];
rz(1.5707963267948966) q[12];
z q[14];
id q[10];
z q[6];
rz(1.5707963267948966) q[4];
id q[6];
id q[8];
u1(1.5707963267948966) q[1];
s q[5];
s q[17];
t q[16];
sdg q[1];
id q[2];
rz(1.5707963267948966) q[12];
tdg q[10];
z q[15];
z q[6];
z q[10];
z q[14];
z q[16];
t q[17];
tdg q[5];
sdg q[12];
id q[5];
z q[5];
t q[4];
z q[12];
rz(1.5707963267948966) q[12];
tdg q[5];
u1(1.5707963267948966) q[13];
id q[2];
id q[10];
z q[16];
