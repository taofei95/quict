OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c[28];
z q[26];
rx(1.5707963267948966) q[17];
h q[18];
h q[27];
rz(1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[9];
x q[22];
u2(1.5707963267948966, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[10];
y q[11];
z q[23];
sdg q[16];
s q[21];
tdg q[11];
rz(1.5707963267948966) q[27];
rz(1.5707963267948966) q[27];
ry(1.5707963267948966) q[14];
s q[10];
u3(0, 0, 1.5707963267948966) q[6];
rx(1.5707963267948966) q[2];
h q[23];
rx(1.5707963267948966) q[24];
t q[8];
rx(1.5707963267948966) q[13];
u1(1.5707963267948966) q[5];
s q[10];
rx(1.5707963267948966) q[12];
h q[22];
u2(1.5707963267948966, 1.5707963267948966) q[25];
x q[3];
rz(1.5707963267948966) q[17];
tdg q[14];
tdg q[16];
t q[9];
u1(1.5707963267948966) q[3];
h q[2];
t q[1];
x q[24];
s q[8];
y q[0];
s q[25];
u3(0, 0, 1.5707963267948966) q[27];
s q[22];
s q[16];
y q[12];
x q[7];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
sdg q[12];
t q[24];
tdg q[10];
x q[11];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[16];
y q[0];
s q[27];
tdg q[25];
sdg q[4];
x q[1];
tdg q[27];
rz(1.5707963267948966) q[0];
sdg q[1];
ry(1.5707963267948966) q[1];
sdg q[13];
x q[5];
u2(1.5707963267948966, 1.5707963267948966) q[23];
u3(0, 0, 1.5707963267948966) q[13];
rz(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[21];
y q[12];
t q[22];
y q[23];
ry(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[7];
s q[5];
sdg q[15];
rz(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[24];
s q[26];
x q[25];
h q[10];
rx(1.5707963267948966) q[13];
h q[20];
s q[19];
z q[15];
rx(1.5707963267948966) q[23];
s q[1];
s q[13];
s q[17];
y q[25];
sdg q[5];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[8];
h q[22];
u2(1.5707963267948966, 1.5707963267948966) q[25];
t q[1];
u3(0, 0, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[9];
x q[26];
x q[2];
rz(1.5707963267948966) q[17];
x q[25];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[13];
z q[4];
z q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[6];
h q[6];
h q[21];
ry(1.5707963267948966) q[19];
sdg q[0];
tdg q[24];
u2(1.5707963267948966, 1.5707963267948966) q[10];
rz(1.5707963267948966) q[20];
rx(1.5707963267948966) q[0];
u1(1.5707963267948966) q[9];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[27];
ry(1.5707963267948966) q[21];
sdg q[21];
rx(1.5707963267948966) q[27];
rx(1.5707963267948966) q[20];
h q[5];
tdg q[27];
sdg q[9];
rz(1.5707963267948966) q[12];
h q[19];
u2(1.5707963267948966, 1.5707963267948966) q[11];
z q[18];
t q[0];
tdg q[17];
z q[19];
s q[20];
rz(1.5707963267948966) q[15];
t q[18];
z q[11];
sdg q[17];
rz(1.5707963267948966) q[26];
s q[5];
t q[27];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[11];
u1(1.5707963267948966) q[24];
x q[7];
s q[17];
t q[0];
tdg q[25];
z q[15];
x q[10];
x q[25];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[6];
t q[11];
z q[26];
rx(1.5707963267948966) q[5];
s q[23];
u1(1.5707963267948966) q[5];
sdg q[4];
sdg q[4];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[19];
h q[5];
rz(1.5707963267948966) q[27];
tdg q[0];
ry(1.5707963267948966) q[24];
y q[10];
tdg q[1];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[8];
s q[24];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[24];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[18];
rx(1.5707963267948966) q[20];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[19];
x q[18];
sdg q[23];
rx(1.5707963267948966) q[18];
u1(1.5707963267948966) q[18];
t q[14];
t q[9];
tdg q[20];
rz(1.5707963267948966) q[2];
h q[6];
x q[3];
y q[9];
h q[17];
u2(1.5707963267948966, 1.5707963267948966) q[10];
x q[9];
u1(1.5707963267948966) q[4];
sdg q[2];
u1(1.5707963267948966) q[17];
h q[18];
t q[13];
ry(1.5707963267948966) q[5];
h q[22];
rz(1.5707963267948966) q[25];
u3(0, 0, 1.5707963267948966) q[13];
z q[3];
y q[24];
u3(0, 0, 1.5707963267948966) q[21];
y q[20];
t q[0];
t q[4];
u3(0, 0, 1.5707963267948966) q[26];
t q[11];
x q[4];
rx(1.5707963267948966) q[27];
y q[22];
t q[15];
t q[15];
h q[22];
x q[9];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[25];
tdg q[14];
h q[16];
sdg q[19];
x q[0];
t q[21];
x q[9];
x q[21];
ry(1.5707963267948966) q[22];
x q[9];
x q[9];
s q[12];
rz(1.5707963267948966) q[7];
y q[7];
rz(1.5707963267948966) q[8];
t q[3];
x q[22];
sdg q[10];
s q[11];
u3(0, 0, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[10];
h q[4];
u3(0, 0, 1.5707963267948966) q[24];
rx(1.5707963267948966) q[26];
h q[1];
rx(1.5707963267948966) q[24];
t q[10];
x q[2];
sdg q[6];
z q[17];
y q[14];
z q[25];
u1(1.5707963267948966) q[15];
u1(1.5707963267948966) q[14];
x q[26];
u1(1.5707963267948966) q[16];
y q[6];
tdg q[24];
u2(1.5707963267948966, 1.5707963267948966) q[10];
rz(1.5707963267948966) q[4];
y q[9];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[1];
x q[22];
x q[18];
ry(1.5707963267948966) q[13];
u1(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[1];
x q[26];
rz(1.5707963267948966) q[13];
sdg q[25];
u1(1.5707963267948966) q[16];
rx(1.5707963267948966) q[27];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[25];
s q[2];
sdg q[3];
x q[19];
x q[4];
x q[27];
x q[0];
s q[24];
u3(0, 0, 1.5707963267948966) q[6];
y q[20];
ry(1.5707963267948966) q[19];
y q[7];
rz(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[23];
y q[13];
u2(1.5707963267948966, 1.5707963267948966) q[21];
z q[12];
z q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
sdg q[14];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[21];
tdg q[4];
y q[20];
s q[10];
y q[11];
rz(1.5707963267948966) q[11];
y q[2];
u1(1.5707963267948966) q[17];
t q[11];
rz(1.5707963267948966) q[27];
x q[3];
z q[11];