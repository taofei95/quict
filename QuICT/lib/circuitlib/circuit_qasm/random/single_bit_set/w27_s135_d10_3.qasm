OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[27];
x q[4];
u2(1.5707963267948966, 1.5707963267948966) q[26];
ry(1.5707963267948966) q[10];
tdg q[0];
u3(0, 0, 1.5707963267948966) q[11];
ry(1.5707963267948966) q[20];
x q[7];
rz(1.5707963267948966) q[15];
s q[1];
sdg q[22];
sdg q[25];
t q[18];
u3(0, 0, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[2];
h q[26];
ry(1.5707963267948966) q[5];
y q[19];
ry(1.5707963267948966) q[22];
s q[23];
rz(1.5707963267948966) q[8];
h q[15];
z q[18];
rz(1.5707963267948966) q[4];
s q[6];
u1(1.5707963267948966) q[7];
s q[20];
z q[21];
t q[13];
sdg q[24];
u1(1.5707963267948966) q[17];
x q[6];
s q[9];
x q[22];
t q[21];
ry(1.5707963267948966) q[22];
h q[8];
h q[23];
t q[8];
sdg q[1];
h q[15];
u1(1.5707963267948966) q[26];
sdg q[9];
s q[20];
u3(0, 0, 1.5707963267948966) q[12];
z q[10];
u1(1.5707963267948966) q[22];
t q[18];
u3(0, 0, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[22];
tdg q[0];
s q[11];
x q[21];
s q[1];
u1(1.5707963267948966) q[22];
z q[10];
t q[12];
y q[11];
sdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
x q[23];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[10];
u1(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[16];
h q[1];
sdg q[8];
ry(1.5707963267948966) q[3];
z q[13];
x q[11];
sdg q[10];
x q[15];
h q[22];
x q[16];
z q[1];
t q[25];
z q[20];
x q[8];
s q[14];
rx(1.5707963267948966) q[17];
tdg q[11];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[14];
tdg q[20];
rx(1.5707963267948966) q[11];
h q[13];
z q[10];
x q[25];
t q[25];
s q[25];
s q[8];
h q[21];
z q[18];
y q[20];
y q[12];
h q[17];
ry(1.5707963267948966) q[18];
ry(1.5707963267948966) q[5];
x q[15];
u1(1.5707963267948966) q[20];
h q[21];
s q[16];
u1(1.5707963267948966) q[23];
ry(1.5707963267948966) q[2];
x q[16];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[11];
u1(1.5707963267948966) q[11];
s q[19];
ry(1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[12];
rz(1.5707963267948966) q[13];
x q[5];
sdg q[1];
sdg q[18];
rz(1.5707963267948966) q[26];
y q[18];
u2(1.5707963267948966, 1.5707963267948966) q[2];
s q[21];
s q[21];
u2(1.5707963267948966, 1.5707963267948966) q[3];
z q[2];
s q[13];
ry(1.5707963267948966) q[26];
z q[16];
u3(0, 0, 1.5707963267948966) q[14];
t q[25];
x q[20];
tdg q[9];
u3(0, 0, 1.5707963267948966) q[26];
h q[11];
tdg q[1];
t q[11];
x q[10];
sdg q[25];
