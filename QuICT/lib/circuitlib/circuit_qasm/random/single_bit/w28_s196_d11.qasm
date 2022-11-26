OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c[28];
u2(1.5707963267948966, 1.5707963267948966) q[24];
u3(0, 0, 1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[7];
y q[7];
x q[26];
z q[7];
s q[2];
h q[16];
sdg q[6];
z q[5];
s q[23];
u1(1.5707963267948966) q[10];
t q[0];
h q[22];
y q[10];
x q[23];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[13];
s q[2];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[8];
h q[12];
rx(1.5707963267948966) q[27];
sdg q[2];
y q[8];
h q[18];
z q[18];
z q[1];
z q[27];
x q[8];
u3(0, 0, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[12];
sdg q[18];
t q[1];
h q[2];
s q[8];
sdg q[8];
sdg q[9];
h q[7];
rx(1.5707963267948966) q[19];
u1(1.5707963267948966) q[26];
tdg q[20];
u1(1.5707963267948966) q[2];
t q[25];
x q[18];
y q[13];
sdg q[27];
h q[17];
x q[24];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[13];
y q[23];
rz(1.5707963267948966) q[18];
sdg q[24];
y q[10];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[11];
z q[23];
z q[11];
tdg q[17];
u2(1.5707963267948966, 1.5707963267948966) q[18];
u1(1.5707963267948966) q[10];
y q[26];
s q[22];
u1(1.5707963267948966) q[2];
x q[21];
y q[4];
u2(1.5707963267948966, 1.5707963267948966) q[23];
z q[10];
z q[17];
rx(1.5707963267948966) q[15];
z q[18];
u3(0, 0, 1.5707963267948966) q[25];
u2(1.5707963267948966, 1.5707963267948966) q[11];
h q[27];
x q[5];
sdg q[2];
y q[2];
h q[22];
rz(1.5707963267948966) q[4];
tdg q[5];
tdg q[7];
u3(0, 0, 1.5707963267948966) q[25];
z q[15];
s q[14];
rx(1.5707963267948966) q[20];
sdg q[8];
x q[12];
u2(1.5707963267948966, 1.5707963267948966) q[26];
x q[9];
rz(1.5707963267948966) q[27];
ry(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[24];
u2(1.5707963267948966, 1.5707963267948966) q[10];
s q[19];
y q[14];
h q[13];
z q[23];
ry(1.5707963267948966) q[12];
y q[13];
u1(1.5707963267948966) q[21];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[20];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[10];
tdg q[18];
sdg q[23];
x q[3];
tdg q[10];
u3(0, 0, 1.5707963267948966) q[27];
t q[7];
tdg q[27];
s q[11];
u3(0, 0, 1.5707963267948966) q[13];
u1(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[17];
h q[27];
y q[9];
y q[22];
x q[5];
y q[25];
y q[13];
u3(0, 0, 1.5707963267948966) q[17];
t q[24];
tdg q[12];
y q[12];
x q[15];
u2(1.5707963267948966, 1.5707963267948966) q[24];
u2(1.5707963267948966, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[16];
h q[15];
u1(1.5707963267948966) q[27];
sdg q[19];
z q[9];
y q[8];
h q[14];
u1(1.5707963267948966) q[6];
s q[21];
tdg q[3];
rx(1.5707963267948966) q[5];
tdg q[0];
z q[18];
h q[26];
u2(1.5707963267948966, 1.5707963267948966) q[7];
s q[1];
x q[13];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[24];
u1(1.5707963267948966) q[18];
y q[14];
tdg q[3];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[15];
u1(1.5707963267948966) q[22];
rz(1.5707963267948966) q[8];
t q[25];
y q[16];
rz(1.5707963267948966) q[6];
y q[5];
tdg q[17];
u3(0, 0, 1.5707963267948966) q[0];
x q[21];
u1(1.5707963267948966) q[14];
u1(1.5707963267948966) q[3];
tdg q[3];
sdg q[22];
u1(1.5707963267948966) q[26];
h q[15];
s q[5];
t q[18];
z q[21];
x q[5];
h q[11];
sdg q[11];
tdg q[2];
z q[20];
tdg q[1];
z q[8];
h q[22];
z q[2];
rz(1.5707963267948966) q[22];
ry(1.5707963267948966) q[11];
sdg q[22];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[22];
sdg q[9];
x q[14];
ry(1.5707963267948966) q[11];
s q[12];
u2(1.5707963267948966, 1.5707963267948966) q[14];
