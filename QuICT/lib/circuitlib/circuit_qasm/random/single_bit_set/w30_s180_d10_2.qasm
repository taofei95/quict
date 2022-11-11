OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
s q[17];
u3(0, 0, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[4];
x q[15];
u2(1.5707963267948966, 1.5707963267948966) q[15];
s q[23];
ry(1.5707963267948966) q[13];
x q[28];
x q[9];
u1(1.5707963267948966) q[26];
s q[11];
s q[20];
h q[3];
ry(1.5707963267948966) q[18];
x q[6];
y q[1];
u1(1.5707963267948966) q[24];
tdg q[3];
u1(1.5707963267948966) q[3];
h q[7];
tdg q[29];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[25];
sdg q[29];
y q[14];
rz(1.5707963267948966) q[24];
ry(1.5707963267948966) q[11];
h q[11];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[0];
y q[20];
z q[15];
tdg q[1];
ry(1.5707963267948966) q[13];
x q[12];
s q[17];
sdg q[12];
ry(1.5707963267948966) q[12];
t q[24];
sdg q[28];
ry(1.5707963267948966) q[25];
tdg q[25];
z q[14];
ry(1.5707963267948966) q[18];
t q[8];
rx(1.5707963267948966) q[3];
x q[18];
u2(1.5707963267948966, 1.5707963267948966) q[12];
h q[19];
sdg q[26];
ry(1.5707963267948966) q[12];
s q[13];
rx(1.5707963267948966) q[22];
ry(1.5707963267948966) q[29];
u1(1.5707963267948966) q[12];
sdg q[14];
rz(1.5707963267948966) q[0];
sdg q[10];
x q[13];
ry(1.5707963267948966) q[3];
y q[14];
y q[4];
u1(1.5707963267948966) q[20];
t q[15];
x q[4];
rx(1.5707963267948966) q[15];
h q[7];
y q[20];
s q[5];
tdg q[27];
y q[5];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[11];
x q[18];
rz(1.5707963267948966) q[8];
y q[25];
y q[2];
s q[12];
u1(1.5707963267948966) q[26];
t q[9];
y q[1];
u2(1.5707963267948966, 1.5707963267948966) q[25];
z q[18];
ry(1.5707963267948966) q[12];
h q[23];
rz(1.5707963267948966) q[22];
sdg q[7];
tdg q[29];
s q[4];
h q[26];
s q[8];
z q[19];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[7];
h q[8];
tdg q[18];
h q[24];
rz(1.5707963267948966) q[16];
s q[5];
t q[18];
rx(1.5707963267948966) q[18];
tdg q[14];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[10];
tdg q[22];
u3(0, 0, 1.5707963267948966) q[1];
s q[7];
u1(1.5707963267948966) q[5];
y q[28];
ry(1.5707963267948966) q[13];
x q[1];
t q[27];
rz(1.5707963267948966) q[29];
u1(1.5707963267948966) q[8];
y q[2];
x q[25];
z q[14];
rz(1.5707963267948966) q[10];
t q[15];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[9];
z q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[10];
h q[13];
rz(1.5707963267948966) q[6];
z q[18];
rx(1.5707963267948966) q[25];
x q[2];
u1(1.5707963267948966) q[24];
rz(1.5707963267948966) q[29];
h q[24];
tdg q[9];
u1(1.5707963267948966) q[9];
u1(1.5707963267948966) q[21];
rx(1.5707963267948966) q[15];
y q[4];
u2(1.5707963267948966, 1.5707963267948966) q[23];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[17];
z q[6];
t q[14];
z q[24];
z q[27];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[18];
x q[9];
u2(1.5707963267948966, 1.5707963267948966) q[22];
s q[6];
u3(0, 0, 1.5707963267948966) q[11];
sdg q[20];
x q[27];
sdg q[28];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[25];
y q[4];
tdg q[8];
z q[12];
sdg q[19];
y q[2];
x q[20];
z q[11];
sdg q[23];
x q[25];
u1(1.5707963267948966) q[26];
rx(1.5707963267948966) q[2];
sdg q[12];
t q[11];
t q[7];
tdg q[26];
rz(1.5707963267948966) q[11];
u1(1.5707963267948966) q[14];
s q[9];
sdg q[25];
s q[1];
rx(1.5707963267948966) q[19];
h q[3];
s q[6];
