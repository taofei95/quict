OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
u1(1.5707963267948966) q[0];
z q[1];
t q[9];
rz(1.5707963267948966) q[7];
z q[1];
s q[14];
t q[6];
s q[12];
sdg q[1];
tdg q[11];
z q[10];
u1(1.5707963267948966) q[12];
id q[14];
sdg q[9];
s q[3];
s q[2];
id q[15];
s q[5];
tdg q[4];
z q[9];
z q[1];
t q[13];
tdg q[13];
z q[14];
t q[15];
id q[2];
t q[8];
rz(1.5707963267948966) q[11];
sdg q[8];
t q[8];
id q[7];
sdg q[9];
sdg q[4];
t q[15];
tdg q[6];
sdg q[14];
tdg q[2];
s q[9];
sdg q[10];
sdg q[11];
tdg q[7];
tdg q[11];
tdg q[12];
z q[15];
z q[1];
t q[2];
z q[0];
u1(1.5707963267948966) q[14];
s q[3];
t q[13];
id q[6];
u1(1.5707963267948966) q[11];
sdg q[10];
s q[3];
tdg q[11];
t q[4];
t q[9];
s q[3];
id q[8];
z q[4];
tdg q[2];
z q[12];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[7];
sdg q[9];
s q[6];
z q[2];
sdg q[14];
t q[5];
s q[4];
id q[0];
t q[9];
tdg q[11];
t q[8];
z q[9];
u1(1.5707963267948966) q[8];
t q[8];
rz(1.5707963267948966) q[0];
sdg q[12];
sdg q[4];
u1(1.5707963267948966) q[11];
tdg q[13];
rz(1.5707963267948966) q[5];
id q[3];
sdg q[0];
sdg q[12];
s q[8];
t q[4];
id q[10];
tdg q[15];
s q[1];
tdg q[11];
u1(1.5707963267948966) q[10];
t q[8];
s q[10];
tdg q[10];
t q[9];
u1(1.5707963267948966) q[3];
sdg q[11];
id q[6];
u1(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[11];
tdg q[11];
s q[0];
rz(1.5707963267948966) q[10];
s q[11];
tdg q[14];
sdg q[5];
sdg q[12];
id q[12];
z q[10];
tdg q[14];
sdg q[15];
t q[14];
s q[8];
u1(1.5707963267948966) q[8];
tdg q[4];
z q[5];
t q[9];
tdg q[6];
z q[7];
id q[1];
rz(1.5707963267948966) q[3];
t q[7];
u1(1.5707963267948966) q[15];
u1(1.5707963267948966) q[6];
s q[1];
