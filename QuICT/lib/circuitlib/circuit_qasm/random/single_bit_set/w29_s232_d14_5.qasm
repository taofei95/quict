OPENQASM 2.0;
include "qelib1.inc";
qreg q[29];
creg c[29];
u3(0, 0, 1.5707963267948966) q[17];
y q[17];
u1(1.5707963267948966) q[26];
x q[6];
rx(1.5707963267948966) q[14];
h q[14];
u2(1.5707963267948966, 1.5707963267948966) q[25];
t q[8];
tdg q[5];
u1(1.5707963267948966) q[12];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[22];
u1(1.5707963267948966) q[25];
y q[19];
u1(1.5707963267948966) q[17];
h q[7];
x q[15];
y q[1];
tdg q[19];
rx(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[22];
rz(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[24];
x q[9];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[24];
rx(1.5707963267948966) q[16];
u1(1.5707963267948966) q[19];
u1(1.5707963267948966) q[28];
s q[8];
sdg q[7];
u1(1.5707963267948966) q[27];
y q[17];
y q[12];
u3(0, 0, 1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[14];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[6];
x q[9];
tdg q[16];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
x q[3];
t q[19];
u1(1.5707963267948966) q[0];
s q[21];
tdg q[2];
tdg q[16];
rz(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[23];
ry(1.5707963267948966) q[12];
t q[11];
y q[5];
t q[27];
x q[20];
s q[6];
u2(1.5707963267948966, 1.5707963267948966) q[22];
sdg q[25];
z q[7];
s q[24];
u1(1.5707963267948966) q[20];
sdg q[22];
sdg q[12];
u1(1.5707963267948966) q[8];
s q[2];
t q[3];
u1(1.5707963267948966) q[1];
sdg q[7];
h q[13];
sdg q[9];
t q[17];
u2(1.5707963267948966, 1.5707963267948966) q[4];
tdg q[19];
z q[4];
u1(1.5707963267948966) q[13];
s q[9];
rx(1.5707963267948966) q[5];
x q[13];
sdg q[2];
rz(1.5707963267948966) q[18];
y q[10];
u2(1.5707963267948966, 1.5707963267948966) q[19];
y q[15];
rx(1.5707963267948966) q[21];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[11];
s q[25];
s q[13];
z q[1];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[4];
z q[25];
x q[25];
h q[2];
rz(1.5707963267948966) q[0];
t q[3];
rx(1.5707963267948966) q[19];
s q[5];
h q[21];
rz(1.5707963267948966) q[1];
tdg q[11];
rx(1.5707963267948966) q[14];
t q[22];
s q[25];
tdg q[0];
tdg q[22];
u3(0, 0, 1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[17];
sdg q[8];
s q[28];
rz(1.5707963267948966) q[14];
y q[4];
t q[28];
s q[5];
y q[23];
t q[22];
u1(1.5707963267948966) q[19];
x q[10];
u2(1.5707963267948966, 1.5707963267948966) q[26];
z q[20];
tdg q[12];
rz(1.5707963267948966) q[1];
x q[10];
x q[13];
tdg q[15];
tdg q[24];
s q[22];
rz(1.5707963267948966) q[5];
s q[1];
x q[4];
z q[15];
s q[10];
u2(1.5707963267948966, 1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[4];
tdg q[18];
sdg q[5];
s q[17];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u1(1.5707963267948966) q[19];
z q[2];
u3(0, 0, 1.5707963267948966) q[23];
tdg q[6];
h q[28];
sdg q[17];
s q[26];
z q[2];
sdg q[28];
h q[16];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[23];
x q[13];
t q[17];
u1(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[27];
t q[1];
rz(1.5707963267948966) q[27];
x q[17];
y q[14];
h q[7];
rx(1.5707963267948966) q[23];
z q[10];
rx(1.5707963267948966) q[4];
x q[14];
t q[20];
rz(1.5707963267948966) q[28];
ry(1.5707963267948966) q[21];
rz(1.5707963267948966) q[27];
rz(1.5707963267948966) q[28];
y q[9];
z q[0];
ry(1.5707963267948966) q[27];
t q[13];
ry(1.5707963267948966) q[23];
sdg q[27];
rz(1.5707963267948966) q[19];
z q[26];
t q[17];
h q[12];
y q[22];
h q[10];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[5];
u1(1.5707963267948966) q[16];
tdg q[14];
z q[24];
ry(1.5707963267948966) q[0];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[9];
s q[2];
h q[27];
tdg q[6];
s q[14];
ry(1.5707963267948966) q[0];
z q[4];
rx(1.5707963267948966) q[14];
rz(1.5707963267948966) q[16];
s q[0];
h q[13];
rz(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[13];
sdg q[20];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[5];
h q[16];
x q[20];
sdg q[18];
rz(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[23];
u3(0, 0, 1.5707963267948966) q[11];
x q[8];
s q[21];
u1(1.5707963267948966) q[15];
u1(1.5707963267948966) q[7];
z q[10];
u1(1.5707963267948966) q[18];
t q[4];
u3(0, 0, 1.5707963267948966) q[0];
h q[13];
u2(1.5707963267948966, 1.5707963267948966) q[4];
sdg q[20];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[6];
x q[9];
rz(1.5707963267948966) q[25];
t q[16];
u3(0, 0, 1.5707963267948966) q[23];
