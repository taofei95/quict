OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[27];
ry(1.5707963267948966) q[20];
x q[23];
ry(1.5707963267948966) q[24];
x q[19];
sdg q[3];
sdg q[19];
tdg q[14];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[14];
rz(1.5707963267948966) q[22];
h q[16];
u3(0, 0, 1.5707963267948966) q[9];
tdg q[15];
u3(0, 0, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[6];
x q[11];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[17];
x q[8];
x q[6];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[11];
rz(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[16];
h q[17];
u3(0, 0, 1.5707963267948966) q[1];
h q[25];
u2(1.5707963267948966, 1.5707963267948966) q[5];
z q[9];
rz(1.5707963267948966) q[4];
y q[23];
z q[25];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
z q[21];
y q[17];
y q[5];
rz(1.5707963267948966) q[23];
u1(1.5707963267948966) q[14];
x q[18];
s q[14];
ry(1.5707963267948966) q[7];
u1(1.5707963267948966) q[15];
y q[19];
z q[3];
u3(0, 0, 1.5707963267948966) q[15];
rz(1.5707963267948966) q[21];
tdg q[16];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[12];
y q[7];
x q[12];
sdg q[20];
u3(0, 0, 1.5707963267948966) q[20];
h q[17];
sdg q[4];
y q[13];
h q[1];
s q[11];
x q[5];
tdg q[19];
u2(1.5707963267948966, 1.5707963267948966) q[24];
y q[13];
u2(1.5707963267948966, 1.5707963267948966) q[0];
y q[0];
h q[6];
rz(1.5707963267948966) q[25];
t q[24];
x q[26];
rx(1.5707963267948966) q[24];
h q[1];
z q[25];
u2(1.5707963267948966, 1.5707963267948966) q[19];
u2(1.5707963267948966, 1.5707963267948966) q[15];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[16];
x q[12];
t q[14];
x q[3];
rx(1.5707963267948966) q[18];
sdg q[11];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[17];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[15];
s q[0];
x q[24];
z q[20];
u1(1.5707963267948966) q[17];
rz(1.5707963267948966) q[26];
u2(1.5707963267948966, 1.5707963267948966) q[14];
u3(0, 0, 1.5707963267948966) q[1];
h q[20];
u3(0, 0, 1.5707963267948966) q[11];
h q[21];
ry(1.5707963267948966) q[8];
s q[11];
u1(1.5707963267948966) q[22];
rz(1.5707963267948966) q[25];
rx(1.5707963267948966) q[13];
z q[23];
rz(1.5707963267948966) q[3];
s q[16];
u2(1.5707963267948966, 1.5707963267948966) q[20];
sdg q[12];
tdg q[4];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[24];
ry(1.5707963267948966) q[21];
rz(1.5707963267948966) q[20];
tdg q[12];
z q[18];
rz(1.5707963267948966) q[5];
t q[16];
u2(1.5707963267948966, 1.5707963267948966) q[1];
s q[1];
sdg q[9];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[24];
sdg q[8];
rx(1.5707963267948966) q[26];
y q[12];
tdg q[15];
x q[16];
tdg q[3];
sdg q[13];
u3(0, 0, 1.5707963267948966) q[22];
z q[24];
x q[7];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[14];
h q[7];
s q[22];
rz(1.5707963267948966) q[20];
s q[25];
h q[24];
u1(1.5707963267948966) q[7];
tdg q[11];
rx(1.5707963267948966) q[6];
x q[21];
z q[4];
rz(1.5707963267948966) q[9];
x q[18];
y q[9];
y q[21];
u2(1.5707963267948966, 1.5707963267948966) q[10];
h q[17];
u2(1.5707963267948966, 1.5707963267948966) q[12];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[24];
sdg q[18];
rz(1.5707963267948966) q[17];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[3];
x q[19];
t q[23];
ry(1.5707963267948966) q[20];
y q[3];
rx(1.5707963267948966) q[24];
u3(0, 0, 1.5707963267948966) q[22];
rz(1.5707963267948966) q[22];
u1(1.5707963267948966) q[17];
u1(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[17];
ry(1.5707963267948966) q[23];
t q[11];
sdg q[9];
t q[26];
tdg q[8];
sdg q[5];
rx(1.5707963267948966) q[4];
u1(1.5707963267948966) q[10];
t q[22];
u2(1.5707963267948966, 1.5707963267948966) q[18];
ry(1.5707963267948966) q[13];
x q[5];
t q[8];
x q[2];
u1(1.5707963267948966) q[15];
sdg q[1];
s q[12];
h q[15];
u3(0, 0, 1.5707963267948966) q[3];
s q[24];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[26];
y q[0];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[24];
u3(0, 0, 1.5707963267948966) q[9];
sdg q[13];
t q[19];
h q[24];
sdg q[2];
u1(1.5707963267948966) q[15];
u1(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[7];
y q[1];
u3(0, 0, 1.5707963267948966) q[23];
x q[16];
u3(0, 0, 1.5707963267948966) q[25];
u2(1.5707963267948966, 1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[25];
u3(0, 0, 1.5707963267948966) q[20];
rz(1.5707963267948966) q[1];
x q[23];
rz(1.5707963267948966) q[7];
z q[7];
u2(1.5707963267948966, 1.5707963267948966) q[3];
y q[5];
t q[0];
u1(1.5707963267948966) q[4];
tdg q[22];
u3(0, 0, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[2];
y q[1];
u3(0, 0, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[8];
y q[26];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
tdg q[24];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[0];
s q[15];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[23];
s q[25];
h q[20];
x q[12];
x q[24];
y q[14];
z q[5];
u1(1.5707963267948966) q[3];
y q[7];
s q[4];
s q[4];
s q[20];
sdg q[10];
tdg q[8];
h q[23];
u1(1.5707963267948966) q[18];
u1(1.5707963267948966) q[0];
sdg q[24];
rx(1.5707963267948966) q[24];
s q[12];
y q[17];
u2(1.5707963267948966, 1.5707963267948966) q[19];
t q[17];
h q[19];
x q[20];
s q[21];
u1(1.5707963267948966) q[0];
s q[22];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[14];
u1(1.5707963267948966) q[13];
y q[7];
sdg q[13];
t q[20];
tdg q[4];
s q[17];
u1(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[6];
t q[11];
sdg q[16];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
z q[3];
rz(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[24];
u2(1.5707963267948966, 1.5707963267948966) q[15];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[18];
ry(1.5707963267948966) q[5];
s q[12];
u1(1.5707963267948966) q[19];
ry(1.5707963267948966) q[9];
t q[2];
rx(1.5707963267948966) q[13];
tdg q[16];
z q[15];
t q[16];
sdg q[17];
sdg q[12];
rx(1.5707963267948966) q[18];
h q[21];
rz(1.5707963267948966) q[22];
y q[26];
h q[24];
s q[13];
t q[25];
y q[22];
rz(1.5707963267948966) q[6];
x q[8];
ry(1.5707963267948966) q[22];
s q[10];
z q[16];
x q[19];
y q[5];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[12];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[23];
t q[2];
z q[23];
x q[12];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[15];
u3(0, 0, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[22];
rx(1.5707963267948966) q[1];
h q[7];
x q[18];
sdg q[26];
u1(1.5707963267948966) q[24];
sdg q[12];
u3(0, 0, 1.5707963267948966) q[4];
tdg q[21];
s q[23];
u3(0, 0, 1.5707963267948966) q[0];
h q[21];
u2(1.5707963267948966, 1.5707963267948966) q[9];
t q[6];
tdg q[21];
tdg q[24];
h q[6];
sdg q[16];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[13];
u1(1.5707963267948966) q[16];
s q[0];
y q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[22];
rz(1.5707963267948966) q[21];
sdg q[5];
sdg q[3];