OPENQASM 2.0;
include "qelib1.inc";
qreg q[29];
creg c[29];
ry(1.5707963267948966) q[14];
u3(0, 0, 1.5707963267948966) q[18];
u1(1.5707963267948966) q[9];
z q[7];
rz(1.5707963267948966) q[17];
tdg q[24];
s q[15];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[12];
x q[20];
rz(1.5707963267948966) q[26];
z q[9];
u1(1.5707963267948966) q[12];
rx(1.5707963267948966) q[22];
rx(1.5707963267948966) q[22];
u3(0, 0, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
t q[15];
h q[13];
s q[26];
sdg q[24];
t q[19];
z q[12];
ry(1.5707963267948966) q[0];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[16];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[24];
t q[16];
s q[14];
y q[28];
u2(1.5707963267948966, 1.5707963267948966) q[22];
rz(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[10];
y q[6];
h q[14];
rz(1.5707963267948966) q[14];
s q[24];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[27];
u1(1.5707963267948966) q[14];
h q[17];
h q[4];
h q[7];
u3(0, 0, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[12];
tdg q[13];
rx(1.5707963267948966) q[5];
u1(1.5707963267948966) q[25];
rx(1.5707963267948966) q[24];
rx(1.5707963267948966) q[1];
y q[16];
z q[11];
t q[15];
y q[1];
u3(0, 0, 1.5707963267948966) q[0];
tdg q[17];
s q[24];
t q[6];
z q[16];
u1(1.5707963267948966) q[10];
x q[3];
t q[0];
u1(1.5707963267948966) q[16];
h q[15];
z q[6];
s q[26];
rx(1.5707963267948966) q[27];
y q[12];
y q[19];
z q[21];
h q[10];
t q[23];
x q[7];
rz(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[8];
x q[21];
rx(1.5707963267948966) q[13];
y q[7];
ry(1.5707963267948966) q[9];
tdg q[2];
ry(1.5707963267948966) q[9];
s q[22];
h q[4];
u3(0, 0, 1.5707963267948966) q[2];
z q[28];
tdg q[19];
rz(1.5707963267948966) q[16];
x q[25];
sdg q[13];
h q[23];
h q[24];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[6];
tdg q[26];
rx(1.5707963267948966) q[8];
h q[7];
y q[7];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u1(1.5707963267948966) q[20];
u2(1.5707963267948966, 1.5707963267948966) q[19];
x q[3];
t q[20];
t q[3];
z q[6];
z q[20];
t q[25];
sdg q[10];
u2(1.5707963267948966, 1.5707963267948966) q[9];
sdg q[18];
u3(0, 0, 1.5707963267948966) q[21];
rz(1.5707963267948966) q[27];
u1(1.5707963267948966) q[11];
rz(1.5707963267948966) q[24];
sdg q[27];
x q[13];
z q[15];
sdg q[11];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[8];
x q[3];
x q[12];
u3(0, 0, 1.5707963267948966) q[22];
z q[0];
y q[1];
y q[2];
u1(1.5707963267948966) q[19];
tdg q[27];
rx(1.5707963267948966) q[20];
h q[21];
t q[27];
y q[21];
tdg q[12];
tdg q[7];
h q[6];
tdg q[16];
u3(0, 0, 1.5707963267948966) q[27];
u3(0, 0, 1.5707963267948966) q[26];
u1(1.5707963267948966) q[28];
u1(1.5707963267948966) q[18];
x q[10];
ry(1.5707963267948966) q[1];
t q[26];
rx(1.5707963267948966) q[13];
y q[21];
rz(1.5707963267948966) q[23];
z q[26];
s q[8];
h q[12];
z q[19];
sdg q[16];
u2(1.5707963267948966, 1.5707963267948966) q[24];
rx(1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[16];
z q[25];
x q[26];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[9];
t q[24];
rz(1.5707963267948966) q[9];
s q[16];
z q[7];
y q[24];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[0];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[26];
rx(1.5707963267948966) q[28];
rx(1.5707963267948966) q[15];
rx(1.5707963267948966) q[13];
z q[20];
sdg q[9];
t q[24];
tdg q[24];
u3(0, 0, 1.5707963267948966) q[4];
h q[9];
u1(1.5707963267948966) q[22];
rz(1.5707963267948966) q[12];
h q[2];
s q[13];
rz(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[26];
u3(0, 0, 1.5707963267948966) q[27];
s q[8];
u3(0, 0, 1.5707963267948966) q[14];
u3(0, 0, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[9];
s q[5];
y q[10];
u1(1.5707963267948966) q[20];
t q[2];
sdg q[12];
x q[21];
u1(1.5707963267948966) q[22];
s q[3];
x q[16];
h q[11];
ry(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[27];
y q[28];