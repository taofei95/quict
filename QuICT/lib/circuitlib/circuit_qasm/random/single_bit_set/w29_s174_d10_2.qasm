OPENQASM 2.0;
include "qelib1.inc";
qreg q[29];
creg c[29];
sdg q[26];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[6];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[21];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[15];
z q[0];
s q[21];
u1(1.5707963267948966) q[4];
t q[26];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[13];
u1(1.5707963267948966) q[28];
ry(1.5707963267948966) q[24];
y q[12];
z q[9];
h q[17];
x q[20];
y q[11];
t q[18];
u2(1.5707963267948966, 1.5707963267948966) q[15];
y q[11];
y q[13];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[13];
x q[4];
sdg q[10];
rx(1.5707963267948966) q[6];
y q[16];
z q[13];
y q[22];
h q[4];
u3(0, 0, 1.5707963267948966) q[24];
rz(1.5707963267948966) q[17];
y q[0];
ry(1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[19];
x q[13];
u3(0, 0, 1.5707963267948966) q[2];
z q[4];
ry(1.5707963267948966) q[3];
h q[1];
ry(1.5707963267948966) q[6];
y q[6];
z q[5];
t q[20];
u3(0, 0, 1.5707963267948966) q[13];
t q[17];
u1(1.5707963267948966) q[10];
u1(1.5707963267948966) q[18];
rz(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[22];
tdg q[5];
tdg q[23];
u3(0, 0, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[27];
x q[7];
s q[14];
rx(1.5707963267948966) q[10];
s q[20];
u2(1.5707963267948966, 1.5707963267948966) q[20];
rz(1.5707963267948966) q[28];
u2(1.5707963267948966, 1.5707963267948966) q[20];
h q[8];
sdg q[20];
sdg q[19];
tdg q[11];
ry(1.5707963267948966) q[27];
h q[27];
x q[4];
rx(1.5707963267948966) q[18];
s q[16];
s q[7];
rz(1.5707963267948966) q[20];
s q[18];
sdg q[8];
z q[16];
y q[7];
y q[8];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[25];
ry(1.5707963267948966) q[24];
u1(1.5707963267948966) q[22];
s q[8];
u3(0, 0, 1.5707963267948966) q[12];
sdg q[5];
u1(1.5707963267948966) q[11];
u1(1.5707963267948966) q[11];
s q[25];
ry(1.5707963267948966) q[26];
h q[21];
t q[11];
u3(0, 0, 1.5707963267948966) q[27];
h q[12];
t q[6];
sdg q[2];
z q[8];
ry(1.5707963267948966) q[7];
sdg q[14];
z q[26];
sdg q[0];
h q[22];
h q[2];
z q[1];
u2(1.5707963267948966, 1.5707963267948966) q[28];
rz(1.5707963267948966) q[22];
y q[14];
x q[14];
t q[4];
s q[10];
rx(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[17];
y q[16];
tdg q[3];
t q[24];
tdg q[14];
t q[16];
x q[16];
t q[19];
tdg q[10];
ry(1.5707963267948966) q[8];
sdg q[25];
u3(0, 0, 1.5707963267948966) q[12];
sdg q[12];
u3(0, 0, 1.5707963267948966) q[24];
rx(1.5707963267948966) q[7];
s q[1];
rx(1.5707963267948966) q[28];
sdg q[26];
x q[3];
sdg q[0];
rz(1.5707963267948966) q[4];
sdg q[17];
sdg q[26];
u2(1.5707963267948966, 1.5707963267948966) q[19];
z q[17];
sdg q[8];
rz(1.5707963267948966) q[1];
x q[0];
tdg q[11];
t q[10];
rx(1.5707963267948966) q[14];
z q[4];
y q[15];
y q[27];
tdg q[28];
u1(1.5707963267948966) q[15];
ry(1.5707963267948966) q[7];
y q[2];
sdg q[2];
rz(1.5707963267948966) q[19];
tdg q[11];
u1(1.5707963267948966) q[23];
tdg q[8];
u1(1.5707963267948966) q[21];
tdg q[13];
rx(1.5707963267948966) q[8];
tdg q[24];
z q[10];
t q[2];
y q[22];
ry(1.5707963267948966) q[23];
rx(1.5707963267948966) q[1];
s q[0];
t q[22];
u1(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[28];
rz(1.5707963267948966) q[9];
h q[2];
y q[20];
y q[17];
ry(1.5707963267948966) q[6];
