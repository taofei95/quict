OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[27];
u3(0, 0, 1.5707963267948966) q[12];
t q[8];
u3(0, 0, 1.5707963267948966) q[23];
t q[1];
u3(0, 0, 1.5707963267948966) q[15];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[13];
h q[21];
s q[24];
rx(1.5707963267948966) q[4];
u1(1.5707963267948966) q[9];
rx(1.5707963267948966) q[8];
y q[17];
sdg q[26];
tdg q[18];
s q[5];
sdg q[6];
h q[19];
sdg q[21];
x q[25];
x q[11];
t q[15];
u3(0, 0, 1.5707963267948966) q[10];
h q[16];
x q[6];
z q[2];
x q[20];
u3(0, 0, 1.5707963267948966) q[23];
rx(1.5707963267948966) q[21];
sdg q[26];
z q[16];
u1(1.5707963267948966) q[7];
s q[8];
sdg q[17];
rx(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[11];
t q[13];
sdg q[17];
s q[1];
s q[9];
sdg q[3];
z q[7];
z q[2];
x q[7];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[9];
x q[15];
sdg q[18];
z q[12];
u2(1.5707963267948966, 1.5707963267948966) q[26];
tdg q[18];
u2(1.5707963267948966, 1.5707963267948966) q[2];
x q[0];
x q[8];
x q[4];
z q[22];
rz(1.5707963267948966) q[10];
x q[12];
tdg q[12];
s q[21];
rz(1.5707963267948966) q[9];
x q[22];
u3(0, 0, 1.5707963267948966) q[18];
z q[22];
s q[9];
ry(1.5707963267948966) q[20];
u1(1.5707963267948966) q[20];
z q[10];
s q[26];
s q[23];
t q[16];
h q[9];
s q[12];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[12];
tdg q[22];
z q[24];
tdg q[7];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[25];
rz(1.5707963267948966) q[23];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[25];
rx(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[6];
u1(1.5707963267948966) q[19];
z q[11];
rx(1.5707963267948966) q[26];
u1(1.5707963267948966) q[6];
t q[0];
rx(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[11];
t q[7];
tdg q[20];
sdg q[0];
x q[3];
u1(1.5707963267948966) q[11];
sdg q[16];
z q[25];
u3(0, 0, 1.5707963267948966) q[19];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[10];
t q[12];
t q[14];
u3(0, 0, 1.5707963267948966) q[15];
rz(1.5707963267948966) q[12];
y q[4];
rz(1.5707963267948966) q[22];
s q[5];
rz(1.5707963267948966) q[21];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[24];
sdg q[7];
rx(1.5707963267948966) q[17];
x q[24];
u2(1.5707963267948966, 1.5707963267948966) q[26];
rz(1.5707963267948966) q[3];
x q[21];
z q[7];
sdg q[23];
z q[2];
h q[7];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[18];
t q[5];
rx(1.5707963267948966) q[25];
x q[15];
s q[5];
s q[12];
h q[11];
u1(1.5707963267948966) q[26];
ry(1.5707963267948966) q[11];
u1(1.5707963267948966) q[19];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[22];
s q[17];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[20];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[10];
y q[26];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[24];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[23];
tdg q[25];
s q[5];
tdg q[15];
rz(1.5707963267948966) q[26];
x q[11];
u3(0, 0, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[7];
z q[2];
t q[24];
x q[12];
u3(0, 0, 1.5707963267948966) q[18];
ry(1.5707963267948966) q[22];
y q[10];
x q[9];
rz(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[9];
ry(1.5707963267948966) q[20];
h q[3];
z q[18];
rx(1.5707963267948966) q[5];
y q[15];
h q[1];
rz(1.5707963267948966) q[2];
sdg q[4];
z q[20];
u1(1.5707963267948966) q[11];
ry(1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[7];
x q[12];
ry(1.5707963267948966) q[16];
t q[24];
z q[18];
s q[1];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[9];
y q[4];
s q[16];
