OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c[28];
tdg q[9];
t q[22];
s q[19];
h q[5];
u2(1.5707963267948966, 1.5707963267948966) q[12];
x q[19];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[21];
y q[4];
h q[5];
z q[15];
t q[26];
u1(1.5707963267948966) q[16];
y q[5];
tdg q[7];
u1(1.5707963267948966) q[3];
t q[3];
rx(1.5707963267948966) q[21];
tdg q[24];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[25];
x q[7];
ry(1.5707963267948966) q[23];
tdg q[10];
h q[2];
z q[16];
tdg q[3];
y q[17];
u1(1.5707963267948966) q[17];
y q[5];
s q[27];
t q[26];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[6];
s q[6];
h q[15];
rx(1.5707963267948966) q[3];
t q[10];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[0];
sdg q[18];
u3(0, 0, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[25];
ry(1.5707963267948966) q[27];
z q[11];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[17];
s q[9];
y q[19];
x q[5];
rz(1.5707963267948966) q[22];
sdg q[12];
ry(1.5707963267948966) q[13];
sdg q[13];
ry(1.5707963267948966) q[3];
sdg q[1];
tdg q[22];
sdg q[14];
rx(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[26];
rx(1.5707963267948966) q[17];
z q[16];
sdg q[20];
u1(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[8];
sdg q[7];
x q[21];
h q[3];
x q[3];
x q[14];
x q[8];
h q[12];
s q[23];
x q[11];
h q[16];
s q[19];
rz(1.5707963267948966) q[23];
x q[26];
t q[12];
t q[21];
rx(1.5707963267948966) q[22];
s q[11];
u3(0, 0, 1.5707963267948966) q[22];
x q[2];
t q[6];
rz(1.5707963267948966) q[13];
s q[6];
rz(1.5707963267948966) q[7];
x q[21];
u3(0, 0, 1.5707963267948966) q[26];
t q[25];
x q[6];
u2(1.5707963267948966, 1.5707963267948966) q[19];
y q[1];
s q[11];
u1(1.5707963267948966) q[0];
t q[9];
sdg q[12];
tdg q[22];
u3(0, 0, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[4];
x q[14];
u3(0, 0, 1.5707963267948966) q[2];
x q[20];
tdg q[12];
tdg q[25];
ry(1.5707963267948966) q[11];
s q[10];
u3(0, 0, 1.5707963267948966) q[1];
x q[16];
u2(1.5707963267948966, 1.5707963267948966) q[18];
x q[21];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[25];
s q[1];
ry(1.5707963267948966) q[27];
u3(0, 0, 1.5707963267948966) q[17];
tdg q[6];
sdg q[0];
ry(1.5707963267948966) q[8];
s q[14];
sdg q[26];
h q[9];
rz(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[22];
z q[21];
x q[0];
ry(1.5707963267948966) q[18];
x q[17];
u1(1.5707963267948966) q[27];
y q[13];
h q[11];
s q[0];
h q[22];
sdg q[27];
s q[27];
s q[22];
rz(1.5707963267948966) q[21];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[16];
z q[22];
tdg q[25];
rx(1.5707963267948966) q[19];
y q[20];
y q[27];
sdg q[11];
tdg q[27];
u1(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[6];
x q[6];
ry(1.5707963267948966) q[24];
tdg q[26];
rz(1.5707963267948966) q[10];
sdg q[6];
sdg q[20];
t q[23];
z q[23];
u1(1.5707963267948966) q[23];
t q[27];
sdg q[11];
sdg q[21];
u1(1.5707963267948966) q[2];
h q[5];
z q[23];
rx(1.5707963267948966) q[3];
sdg q[14];
s q[14];
ry(1.5707963267948966) q[15];
z q[16];
u1(1.5707963267948966) q[27];
z q[19];
s q[18];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[10];
h q[11];
sdg q[20];
tdg q[26];
y q[22];
y q[9];
rx(1.5707963267948966) q[3];
s q[4];
h q[0];
u1(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[18];
t q[2];
u1(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[9];
h q[1];
s q[0];
u3(0, 0, 1.5707963267948966) q[19];
u2(1.5707963267948966, 1.5707963267948966) q[22];
tdg q[10];
z q[2];
u3(0, 0, 1.5707963267948966) q[16];
tdg q[24];
rx(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[14];
rz(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[13];
rz(1.5707963267948966) q[21];
tdg q[14];
sdg q[6];
y q[22];
tdg q[5];
s q[26];
ry(1.5707963267948966) q[12];
tdg q[2];
x q[11];
rx(1.5707963267948966) q[26];
y q[23];
u1(1.5707963267948966) q[23];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[24];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[27];
tdg q[7];
y q[17];
z q[8];
u3(0, 0, 1.5707963267948966) q[13];
s q[23];
ry(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[26];
x q[11];
u2(1.5707963267948966, 1.5707963267948966) q[22];
sdg q[27];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[0];
t q[0];
u1(1.5707963267948966) q[25];
u3(0, 0, 1.5707963267948966) q[15];
s q[17];
u3(0, 0, 1.5707963267948966) q[22];
t q[9];
u1(1.5707963267948966) q[17];
tdg q[10];
x q[24];
rx(1.5707963267948966) q[17];
tdg q[15];
tdg q[11];
ry(1.5707963267948966) q[2];
u1(1.5707963267948966) q[26];
u1(1.5707963267948966) q[11];
rx(1.5707963267948966) q[20];
z q[16];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[11];
sdg q[4];
rx(1.5707963267948966) q[23];
u2(1.5707963267948966, 1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[23];
y q[22];
h q[18];
t q[23];
z q[1];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[23];
y q[6];
u3(0, 0, 1.5707963267948966) q[20];
rz(1.5707963267948966) q[16];
z q[7];
tdg q[14];
y q[1];
u2(1.5707963267948966, 1.5707963267948966) q[25];
h q[12];
y q[10];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[11];
ry(1.5707963267948966) q[3];
t q[19];
h q[16];
s q[15];
s q[6];
y q[18];
s q[9];
sdg q[10];
z q[6];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[17];
ry(1.5707963267948966) q[5];
s q[6];
x q[27];
u3(0, 0, 1.5707963267948966) q[25];
z q[17];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[23];
sdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[19];
rz(1.5707963267948966) q[9];
z q[11];
sdg q[3];
rz(1.5707963267948966) q[15];
s q[5];
rz(1.5707963267948966) q[25];
sdg q[5];
z q[6];
z q[10];
z q[10];
u2(1.5707963267948966, 1.5707963267948966) q[24];
sdg q[1];
x q[11];
rx(1.5707963267948966) q[16];
tdg q[25];
tdg q[19];
z q[5];
sdg q[5];
sdg q[13];
u3(0, 0, 1.5707963267948966) q[22];
y q[23];
rx(1.5707963267948966) q[22];
u1(1.5707963267948966) q[8];
tdg q[7];
x q[26];
rx(1.5707963267948966) q[17];
z q[4];
ry(1.5707963267948966) q[19];
h q[22];
y q[13];
rx(1.5707963267948966) q[4];
s q[10];
y q[16];
z q[11];
h q[19];
u2(1.5707963267948966, 1.5707963267948966) q[26];
u1(1.5707963267948966) q[17];
u1(1.5707963267948966) q[24];
sdg q[25];
ry(1.5707963267948966) q[16];
t q[20];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[13];
s q[0];
z q[21];
rz(1.5707963267948966) q[26];
rx(1.5707963267948966) q[23];
tdg q[12];
s q[7];
t q[23];
y q[0];
tdg q[16];
rz(1.5707963267948966) q[9];
x q[11];
rz(1.5707963267948966) q[23];
u2(1.5707963267948966, 1.5707963267948966) q[11];
y q[23];
y q[20];
tdg q[14];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[11];
z q[5];
z q[5];
u1(1.5707963267948966) q[11];
t q[21];
z q[24];
rz(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[13];
sdg q[20];
rz(1.5707963267948966) q[19];
u2(1.5707963267948966, 1.5707963267948966) q[5];
s q[16];
z q[11];
h q[20];
sdg q[9];
h q[0];
s q[6];
sdg q[7];
rx(1.5707963267948966) q[26];
tdg q[26];
t q[7];
rx(1.5707963267948966) q[23];
s q[25];
t q[0];
ry(1.5707963267948966) q[14];
h q[21];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[26];
t q[9];
s q[20];
rz(1.5707963267948966) q[12];
s q[8];
s q[1];
rx(1.5707963267948966) q[19];
