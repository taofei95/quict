OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c[28];
u1(1.5707963267948966) q[17];
s q[17];
h q[7];
tdg q[17];
rz(1.5707963267948966) q[20];
ry(1.5707963267948966) q[23];
tdg q[14];
x q[0];
ry(1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[27];
u1(1.5707963267948966) q[27];
x q[8];
z q[25];
z q[4];
y q[19];
ry(1.5707963267948966) q[27];
rx(1.5707963267948966) q[5];
y q[26];
h q[21];
sdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[10];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[8];
sdg q[3];
u1(1.5707963267948966) q[25];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[2];
z q[3];
h q[25];
h q[11];
u3(0, 0, 1.5707963267948966) q[23];
u3(0, 0, 1.5707963267948966) q[18];
h q[14];
ry(1.5707963267948966) q[3];
sdg q[22];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[1];
z q[17];
z q[16];
ry(1.5707963267948966) q[8];
y q[6];
s q[3];
h q[0];
sdg q[7];
rz(1.5707963267948966) q[10];
h q[24];
t q[0];
h q[18];
rx(1.5707963267948966) q[23];
rx(1.5707963267948966) q[13];
tdg q[22];
s q[22];
u3(0, 0, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[19];
sdg q[14];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[6];
sdg q[6];
s q[9];
u1(1.5707963267948966) q[23];
y q[19];
x q[1];
h q[5];
u2(1.5707963267948966, 1.5707963267948966) q[10];
h q[14];
u3(0, 0, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
z q[16];
u3(0, 0, 1.5707963267948966) q[7];
s q[19];
s q[14];
z q[1];
t q[13];
z q[7];
t q[6];
tdg q[20];
ry(1.5707963267948966) q[16];
u1(1.5707963267948966) q[26];
z q[27];
u2(1.5707963267948966, 1.5707963267948966) q[26];
ry(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[4];
h q[3];
s q[19];
y q[6];
t q[7];
rx(1.5707963267948966) q[13];
s q[1];
x q[4];
sdg q[8];
sdg q[25];
u1(1.5707963267948966) q[3];
t q[21];
ry(1.5707963267948966) q[4];
s q[27];
h q[2];
y q[0];
u3(0, 0, 1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u1(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[2];
z q[8];
ry(1.5707963267948966) q[20];
tdg q[4];
s q[18];
y q[27];
u3(0, 0, 1.5707963267948966) q[24];
rz(1.5707963267948966) q[4];
z q[11];
z q[8];
t q[13];
t q[26];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[10];
z q[6];
y q[13];
u3(0, 0, 1.5707963267948966) q[14];
s q[9];
z q[0];
u3(0, 0, 1.5707963267948966) q[10];
rz(1.5707963267948966) q[19];
x q[1];
rz(1.5707963267948966) q[25];
sdg q[7];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[7];
s q[8];
u3(0, 0, 1.5707963267948966) q[24];
t q[1];
tdg q[9];
x q[12];
sdg q[16];
h q[13];
rz(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[1];
y q[10];
u2(1.5707963267948966, 1.5707963267948966) q[18];
u3(0, 0, 1.5707963267948966) q[5];
t q[4];
h q[17];
u3(0, 0, 1.5707963267948966) q[7];
rz(1.5707963267948966) q[4];
x q[16];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[4];
z q[25];
y q[16];
s q[26];
z q[25];
s q[1];
ry(1.5707963267948966) q[3];
tdg q[1];
h q[1];
s q[26];
u3(0, 0, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[6];
rz(1.5707963267948966) q[14];
z q[5];
x q[4];
t q[24];
u1(1.5707963267948966) q[26];
u3(0, 0, 1.5707963267948966) q[26];
t q[10];
y q[21];
y q[13];
rz(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[24];
u2(1.5707963267948966, 1.5707963267948966) q[2];
z q[5];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u1(1.5707963267948966) q[11];
rz(1.5707963267948966) q[27];
z q[8];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[24];
z q[7];
h q[1];
rx(1.5707963267948966) q[2];
x q[26];
s q[12];
ry(1.5707963267948966) q[9];
tdg q[9];
x q[3];
sdg q[25];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[26];
tdg q[24];
h q[25];
rx(1.5707963267948966) q[5];
s q[4];
ry(1.5707963267948966) q[20];
y q[13];
s q[19];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[21];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[12];
y q[2];
rz(1.5707963267948966) q[26];
y q[0];
rx(1.5707963267948966) q[7];
x q[12];
u1(1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[4];
s q[23];
h q[18];
sdg q[12];
sdg q[4];
sdg q[5];
z q[26];
sdg q[26];
sdg q[0];
s q[22];
s q[2];
tdg q[7];
z q[25];
x q[6];
s q[11];
rx(1.5707963267948966) q[4];
x q[9];
sdg q[14];
rz(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[18];
z q[9];
tdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[17];
h q[6];
s q[13];
h q[24];
rx(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[23];
u3(0, 0, 1.5707963267948966) q[12];
ry(1.5707963267948966) q[22];
tdg q[1];
rx(1.5707963267948966) q[0];
u1(1.5707963267948966) q[18];
u1(1.5707963267948966) q[11];
tdg q[13];
rx(1.5707963267948966) q[14];
h q[7];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[13];
sdg q[9];
x q[7];
u3(0, 0, 1.5707963267948966) q[17];
rx(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[24];
t q[18];
u2(1.5707963267948966, 1.5707963267948966) q[15];
y q[25];
s q[27];
z q[12];
ry(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[20];
t q[27];
t q[20];
rz(1.5707963267948966) q[13];
t q[12];
u1(1.5707963267948966) q[12];
x q[23];
z q[9];
u1(1.5707963267948966) q[24];
tdg q[12];
rx(1.5707963267948966) q[4];
y q[15];
z q[17];
rz(1.5707963267948966) q[22];
z q[7];
ry(1.5707963267948966) q[9];
u1(1.5707963267948966) q[7];
tdg q[18];
rz(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[24];
s q[12];
rx(1.5707963267948966) q[16];
s q[22];
y q[2];
sdg q[27];
s q[24];
rz(1.5707963267948966) q[23];
s q[3];
tdg q[16];
s q[21];
z q[9];
u1(1.5707963267948966) q[5];
s q[7];
ry(1.5707963267948966) q[17];
y q[11];
z q[4];
rz(1.5707963267948966) q[15];
x q[7];
sdg q[3];
rz(1.5707963267948966) q[8];
t q[9];
s q[2];
u1(1.5707963267948966) q[11];
x q[21];
sdg q[5];
tdg q[12];
z q[19];
u3(0, 0, 1.5707963267948966) q[10];
h q[16];
tdg q[16];
u1(1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[0];
s q[5];
sdg q[26];
h q[17];
ry(1.5707963267948966) q[18];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[21];
ry(1.5707963267948966) q[25];
ry(1.5707963267948966) q[23];
x q[5];
z q[3];
y q[10];
rx(1.5707963267948966) q[10];
y q[21];
h q[1];
z q[27];
s q[24];
s q[0];
ry(1.5707963267948966) q[7];
z q[10];
h q[1];
tdg q[12];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[22];
y q[24];
y q[22];
u3(0, 0, 1.5707963267948966) q[19];
z q[27];
u3(0, 0, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[21];
tdg q[6];
t q[0];
z q[10];
u1(1.5707963267948966) q[21];
ry(1.5707963267948966) q[25];
x q[12];
h q[5];
x q[27];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[5];
h q[13];
u3(0, 0, 1.5707963267948966) q[23];
h q[26];
rz(1.5707963267948966) q[5];
t q[21];
h q[12];
y q[23];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[8];
sdg q[10];
x q[3];
tdg q[19];
ry(1.5707963267948966) q[11];
z q[21];
z q[26];
x q[24];
ry(1.5707963267948966) q[19];
t q[15];
s q[10];
u1(1.5707963267948966) q[13];
sdg q[16];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[25];
y q[27];
u2(1.5707963267948966, 1.5707963267948966) q[10];
x q[7];
y q[9];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[20];
t q[2];
y q[10];
x q[6];
tdg q[10];
rx(1.5707963267948966) q[21];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[8];
z q[11];
t q[14];
z q[13];
