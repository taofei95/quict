OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
ry(1.5707963267948966) q[19];
u1(1.5707963267948966) q[21];
rz(1.5707963267948966) q[19];
tdg q[20];
t q[8];
h q[1];
x q[13];
s q[5];
u1(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[13];
rz(1.5707963267948966) q[23];
z q[24];
z q[27];
y q[1];
rx(1.5707963267948966) q[19];
rz(1.5707963267948966) q[0];
tdg q[15];
s q[29];
y q[17];
s q[3];
s q[22];
u2(1.5707963267948966, 1.5707963267948966) q[29];
rz(1.5707963267948966) q[19];
h q[23];
rx(1.5707963267948966) q[25];
rz(1.5707963267948966) q[5];
y q[27];
x q[9];
u1(1.5707963267948966) q[6];
t q[16];
x q[0];
u1(1.5707963267948966) q[12];
t q[16];
h q[15];
sdg q[27];
t q[10];
x q[16];
u3(0, 0, 1.5707963267948966) q[16];
ry(1.5707963267948966) q[25];
h q[16];
u2(1.5707963267948966, 1.5707963267948966) q[26];
sdg q[15];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[24];
y q[26];
y q[17];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[29];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[8];
h q[11];
z q[10];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[8];
x q[25];
h q[7];
sdg q[13];
t q[27];
ry(1.5707963267948966) q[24];
z q[5];
ry(1.5707963267948966) q[24];
rz(1.5707963267948966) q[23];
x q[22];
tdg q[25];
rz(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[18];
h q[21];
y q[28];
y q[25];
rx(1.5707963267948966) q[7];
sdg q[26];
u3(0, 0, 1.5707963267948966) q[25];
sdg q[19];
t q[21];
rx(1.5707963267948966) q[11];
h q[0];
h q[9];
s q[20];
t q[14];
y q[20];
y q[25];
x q[14];
rx(1.5707963267948966) q[19];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[29];
rz(1.5707963267948966) q[1];
t q[29];
y q[26];
tdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[10];
x q[10];
s q[20];
u2(1.5707963267948966, 1.5707963267948966) q[17];
rz(1.5707963267948966) q[21];
ry(1.5707963267948966) q[17];
tdg q[7];
t q[15];
ry(1.5707963267948966) q[11];
y q[23];
tdg q[27];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[3];
tdg q[19];
s q[4];
s q[8];
z q[7];
z q[10];
h q[15];
tdg q[10];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[8];
y q[10];
tdg q[16];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[22];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[14];
z q[11];
u2(1.5707963267948966, 1.5707963267948966) q[20];
t q[13];
x q[23];
x q[7];
rx(1.5707963267948966) q[9];
y q[18];
tdg q[26];
t q[6];
ry(1.5707963267948966) q[23];
h q[18];
y q[2];
y q[13];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[29];
u2(1.5707963267948966, 1.5707963267948966) q[26];
x q[18];
u3(0, 0, 1.5707963267948966) q[4];
y q[18];
x q[18];
u2(1.5707963267948966, 1.5707963267948966) q[7];
tdg q[6];
rx(1.5707963267948966) q[15];
rx(1.5707963267948966) q[13];
h q[1];
x q[23];
u2(1.5707963267948966, 1.5707963267948966) q[22];
u2(1.5707963267948966, 1.5707963267948966) q[25];
u1(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[9];
t q[0];
u2(1.5707963267948966, 1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[25];
rz(1.5707963267948966) q[19];
u1(1.5707963267948966) q[10];
y q[24];
y q[18];
h q[25];
u2(1.5707963267948966, 1.5707963267948966) q[29];
tdg q[18];
u3(0, 0, 1.5707963267948966) q[24];
h q[16];
u3(0, 0, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[29];
rx(1.5707963267948966) q[7];
s q[17];
x q[28];
s q[16];
u1(1.5707963267948966) q[16];
sdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[28];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[15];
sdg q[25];
sdg q[29];
s q[8];
sdg q[19];
s q[23];
sdg q[18];
s q[6];
rz(1.5707963267948966) q[9];
z q[8];
t q[13];
rx(1.5707963267948966) q[3];
z q[0];
u1(1.5707963267948966) q[7];
sdg q[16];
rx(1.5707963267948966) q[11];
t q[3];
ry(1.5707963267948966) q[26];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[23];
t q[26];
h q[11];
u2(1.5707963267948966, 1.5707963267948966) q[13];
y q[0];
t q[18];
rx(1.5707963267948966) q[16];
t q[23];
u2(1.5707963267948966, 1.5707963267948966) q[28];
u2(1.5707963267948966, 1.5707963267948966) q[7];
x q[23];
u1(1.5707963267948966) q[26];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[18];
s q[12];
u3(0, 0, 1.5707963267948966) q[22];
h q[17];
