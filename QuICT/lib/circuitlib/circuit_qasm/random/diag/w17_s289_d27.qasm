OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
tdg q[1];
tdg q[10];
rz(1.5707963267948966) q[15];
t q[11];
z q[7];
t q[1];
sdg q[15];
z q[11];
z q[12];
tdg q[6];
tdg q[7];
rz(1.5707963267948966) q[13];
s q[16];
z q[6];
s q[15];
t q[3];
id q[9];
tdg q[4];
s q[10];
sdg q[11];
sdg q[7];
u1(1.5707963267948966) q[16];
z q[14];
tdg q[16];
s q[6];
z q[4];
sdg q[5];
id q[14];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[3];
z q[1];
tdg q[13];
tdg q[6];
s q[0];
rz(1.5707963267948966) q[14];
u1(1.5707963267948966) q[12];
u1(1.5707963267948966) q[4];
s q[2];
s q[12];
u1(1.5707963267948966) q[8];
id q[8];
u1(1.5707963267948966) q[16];
u1(1.5707963267948966) q[4];
s q[8];
t q[1];
t q[16];
id q[14];
rz(1.5707963267948966) q[13];
t q[13];
s q[9];
sdg q[3];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[10];
u1(1.5707963267948966) q[1];
t q[14];
t q[11];
tdg q[10];
rz(1.5707963267948966) q[7];
tdg q[10];
t q[3];
z q[5];
rz(1.5707963267948966) q[3];
sdg q[15];
id q[2];
id q[4];
sdg q[6];
tdg q[13];
t q[13];
u1(1.5707963267948966) q[14];
t q[2];
tdg q[6];
sdg q[9];
u1(1.5707963267948966) q[9];
sdg q[9];
tdg q[6];
s q[6];
t q[2];
t q[12];
sdg q[14];
id q[3];
tdg q[11];
t q[13];
sdg q[2];
s q[6];
t q[11];
z q[15];
s q[9];
id q[3];
id q[12];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[0];
t q[10];
sdg q[6];
id q[9];
z q[16];
rz(1.5707963267948966) q[7];
id q[13];
s q[7];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[7];
s q[16];
u1(1.5707963267948966) q[0];
sdg q[16];
s q[14];
u1(1.5707963267948966) q[15];
z q[15];
sdg q[5];
z q[1];
id q[9];
t q[2];
u1(1.5707963267948966) q[6];
id q[0];
t q[15];
u1(1.5707963267948966) q[3];
tdg q[3];
rz(1.5707963267948966) q[10];
id q[1];
s q[11];
t q[6];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[6];
s q[6];
id q[14];
u1(1.5707963267948966) q[16];
u1(1.5707963267948966) q[4];
tdg q[4];
s q[16];
id q[3];
sdg q[3];
z q[10];
rz(1.5707963267948966) q[3];
t q[9];
sdg q[6];
rz(1.5707963267948966) q[7];
tdg q[14];
z q[14];
z q[8];
id q[10];
u1(1.5707963267948966) q[14];
rz(1.5707963267948966) q[14];
tdg q[10];
z q[4];
id q[9];
id q[6];
sdg q[6];
u1(1.5707963267948966) q[4];
id q[4];
sdg q[7];
sdg q[14];
s q[7];
t q[4];
s q[2];
t q[10];
z q[4];
id q[8];
u1(1.5707963267948966) q[14];
sdg q[15];
tdg q[10];
u1(1.5707963267948966) q[9];
rz(1.5707963267948966) q[1];
s q[11];
id q[14];
tdg q[13];
u1(1.5707963267948966) q[7];
id q[15];
rz(1.5707963267948966) q[15];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[14];
tdg q[7];
sdg q[8];
t q[5];
t q[8];
sdg q[12];
z q[8];
z q[9];
sdg q[0];
rz(1.5707963267948966) q[16];
id q[1];
s q[6];
id q[2];
rz(1.5707963267948966) q[16];
sdg q[14];
id q[6];
z q[4];
z q[8];
u1(1.5707963267948966) q[13];
tdg q[16];
sdg q[1];
tdg q[0];
t q[5];
u1(1.5707963267948966) q[14];
z q[15];
tdg q[9];
t q[5];
t q[13];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[12];
s q[15];
z q[9];
u1(1.5707963267948966) q[6];
z q[14];
tdg q[4];
rz(1.5707963267948966) q[6];
s q[13];
sdg q[4];
rz(1.5707963267948966) q[9];
sdg q[8];
id q[14];
s q[7];
t q[5];
tdg q[7];
t q[16];
tdg q[14];
z q[15];
u1(1.5707963267948966) q[11];
s q[14];
id q[9];
z q[1];
u1(1.5707963267948966) q[11];
sdg q[0];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[0];
z q[7];
z q[2];
sdg q[14];
s q[12];
tdg q[0];
id q[5];
u1(1.5707963267948966) q[7];
sdg q[16];
id q[16];
id q[10];
z q[15];
sdg q[12];
id q[9];
tdg q[14];
sdg q[2];
z q[2];
s q[2];
id q[2];
z q[6];
sdg q[0];
z q[11];
s q[4];
s q[2];
z q[11];
id q[11];
z q[7];
t q[13];
rz(1.5707963267948966) q[3];
z q[2];
rz(1.5707963267948966) q[13];
s q[8];
id q[8];
rz(1.5707963267948966) q[1];
z q[7];
sdg q[6];
sdg q[4];
z q[1];
rz(1.5707963267948966) q[13];
tdg q[6];
u1(1.5707963267948966) q[6];
t q[1];
s q[10];
tdg q[5];
u1(1.5707963267948966) q[13];
rz(1.5707963267948966) q[5];
z q[0];
rz(1.5707963267948966) q[11];
sdg q[4];
z q[14];
rz(1.5707963267948966) q[13];
s q[10];
tdg q[0];
tdg q[11];
z q[5];
z q[3];
t q[11];
u1(1.5707963267948966) q[6];
t q[6];
u1(1.5707963267948966) q[13];
t q[1];
id q[0];
z q[1];
z q[5];
s q[0];
rz(1.5707963267948966) q[16];
s q[15];
