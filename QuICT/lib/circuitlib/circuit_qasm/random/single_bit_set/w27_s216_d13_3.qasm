OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[27];
h q[8];
z q[19];
u1(1.5707963267948966) q[23];
u3(0, 0, 1.5707963267948966) q[7];
t q[1];
u2(1.5707963267948966, 1.5707963267948966) q[21];
z q[0];
u1(1.5707963267948966) q[21];
u2(1.5707963267948966, 1.5707963267948966) q[0];
tdg q[13];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[14];
x q[21];
y q[17];
rz(1.5707963267948966) q[1];
y q[26];
x q[19];
sdg q[6];
u3(0, 0, 1.5707963267948966) q[25];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
y q[25];
z q[19];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[21];
ry(1.5707963267948966) q[24];
s q[1];
rx(1.5707963267948966) q[1];
z q[7];
rx(1.5707963267948966) q[19];
sdg q[18];
h q[25];
u1(1.5707963267948966) q[25];
sdg q[4];
x q[1];
u1(1.5707963267948966) q[7];
x q[23];
ry(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[10];
tdg q[8];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[19];
y q[8];
rx(1.5707963267948966) q[16];
u1(1.5707963267948966) q[17];
rx(1.5707963267948966) q[12];
y q[4];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[6];
x q[19];
rz(1.5707963267948966) q[21];
s q[20];
rz(1.5707963267948966) q[18];
sdg q[8];
z q[25];
u1(1.5707963267948966) q[15];
z q[14];
ry(1.5707963267948966) q[26];
t q[12];
t q[14];
u3(0, 0, 1.5707963267948966) q[6];
x q[6];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[15];
u1(1.5707963267948966) q[25];
rx(1.5707963267948966) q[23];
t q[6];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[23];
tdg q[16];
rz(1.5707963267948966) q[15];
tdg q[19];
s q[9];
rx(1.5707963267948966) q[25];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[12];
z q[4];
rx(1.5707963267948966) q[22];
sdg q[19];
rz(1.5707963267948966) q[26];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
s q[26];
ry(1.5707963267948966) q[24];
h q[14];
u3(0, 0, 1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[6];
tdg q[15];
y q[21];
u2(1.5707963267948966, 1.5707963267948966) q[7];
x q[14];
tdg q[17];
z q[13];
x q[3];
x q[19];
h q[10];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[23];
tdg q[24];
h q[17];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[5];
x q[25];
h q[13];
t q[20];
y q[4];
z q[12];
h q[13];
y q[12];
h q[23];
u1(1.5707963267948966) q[22];
y q[10];
u2(1.5707963267948966, 1.5707963267948966) q[23];
sdg q[18];
u3(0, 0, 1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[26];
x q[16];
y q[24];
h q[2];
u3(0, 0, 1.5707963267948966) q[19];
rz(1.5707963267948966) q[1];
t q[5];
u1(1.5707963267948966) q[20];
tdg q[11];
rx(1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[24];
h q[4];
rx(1.5707963267948966) q[13];
x q[24];
tdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[23];
x q[26];
y q[0];
y q[6];
rx(1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[23];
h q[20];
tdg q[1];
ry(1.5707963267948966) q[21];
u1(1.5707963267948966) q[22];
s q[16];
t q[22];
u1(1.5707963267948966) q[10];
x q[25];
h q[10];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[17];
t q[19];
ry(1.5707963267948966) q[14];
y q[14];
rx(1.5707963267948966) q[14];
t q[26];
u1(1.5707963267948966) q[13];
x q[19];
h q[18];
sdg q[5];
rz(1.5707963267948966) q[26];
u1(1.5707963267948966) q[1];
tdg q[9];
sdg q[6];
y q[8];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[20];
rz(1.5707963267948966) q[14];
u1(1.5707963267948966) q[11];
t q[18];
s q[22];
u2(1.5707963267948966, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[21];
tdg q[18];
u1(1.5707963267948966) q[18];
rz(1.5707963267948966) q[23];
h q[6];
s q[12];
x q[23];
rz(1.5707963267948966) q[23];
tdg q[16];
ry(1.5707963267948966) q[9];
s q[19];
ry(1.5707963267948966) q[25];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[20];
t q[24];
t q[4];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[4];
z q[22];
tdg q[17];
u2(1.5707963267948966, 1.5707963267948966) q[18];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[1];
sdg q[26];
y q[24];
tdg q[21];
rz(1.5707963267948966) q[26];
z q[16];
rx(1.5707963267948966) q[8];
sdg q[5];
tdg q[0];
s q[24];
ry(1.5707963267948966) q[8];
y q[13];
ry(1.5707963267948966) q[13];
h q[12];
rz(1.5707963267948966) q[16];
sdg q[20];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
y q[17];
