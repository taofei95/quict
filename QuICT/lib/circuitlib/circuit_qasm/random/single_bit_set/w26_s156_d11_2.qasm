OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
creg c[26];
u1(1.5707963267948966) q[12];
t q[12];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[22];
ry(1.5707963267948966) q[7];
t q[21];
sdg q[21];
tdg q[17];
z q[16];
ry(1.5707963267948966) q[24];
s q[4];
ry(1.5707963267948966) q[20];
t q[7];
y q[18];
ry(1.5707963267948966) q[25];
ry(1.5707963267948966) q[24];
u1(1.5707963267948966) q[20];
x q[9];
u1(1.5707963267948966) q[14];
sdg q[9];
s q[20];
u2(1.5707963267948966, 1.5707963267948966) q[19];
sdg q[10];
s q[7];
tdg q[0];
tdg q[4];
y q[10];
z q[19];
tdg q[12];
t q[18];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[6];
z q[17];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[24];
u1(1.5707963267948966) q[22];
u2(1.5707963267948966, 1.5707963267948966) q[7];
h q[20];
t q[20];
tdg q[8];
sdg q[2];
rz(1.5707963267948966) q[20];
s q[18];
y q[1];
u1(1.5707963267948966) q[7];
h q[2];
u3(0, 0, 1.5707963267948966) q[23];
s q[23];
rx(1.5707963267948966) q[15];
z q[22];
t q[12];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u1(1.5707963267948966) q[20];
tdg q[3];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[10];
t q[1];
s q[9];
tdg q[19];
z q[25];
rz(1.5707963267948966) q[14];
sdg q[1];
h q[5];
u1(1.5707963267948966) q[10];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[10];
sdg q[5];
h q[20];
u3(0, 0, 1.5707963267948966) q[19];
sdg q[20];
ry(1.5707963267948966) q[25];
rz(1.5707963267948966) q[3];
h q[12];
tdg q[15];
u2(1.5707963267948966, 1.5707963267948966) q[14];
sdg q[18];
u1(1.5707963267948966) q[24];
tdg q[17];
t q[2];
t q[21];
ry(1.5707963267948966) q[23];
u3(0, 0, 1.5707963267948966) q[25];
y q[21];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[15];
h q[6];
x q[5];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[19];
sdg q[8];
t q[9];
u2(1.5707963267948966, 1.5707963267948966) q[19];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[2];
y q[17];
s q[3];
u1(1.5707963267948966) q[12];
h q[19];
t q[18];
x q[19];
u1(1.5707963267948966) q[6];
z q[4];
x q[15];
u2(1.5707963267948966, 1.5707963267948966) q[16];
x q[22];
sdg q[2];
tdg q[25];
sdg q[16];
t q[2];
u1(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[21];
sdg q[7];
y q[0];
y q[13];
h q[21];
rz(1.5707963267948966) q[11];
h q[6];
z q[11];
rz(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[16];
t q[20];
u3(0, 0, 1.5707963267948966) q[12];
y q[21];
x q[5];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[19];
h q[25];
ry(1.5707963267948966) q[21];
u1(1.5707963267948966) q[12];
u1(1.5707963267948966) q[24];
z q[16];
ry(1.5707963267948966) q[19];
u2(1.5707963267948966, 1.5707963267948966) q[17];
ry(1.5707963267948966) q[10];
y q[8];
tdg q[15];
s q[10];
z q[11];
u3(0, 0, 1.5707963267948966) q[18];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[15];
rx(1.5707963267948966) q[15];
x q[4];
u3(0, 0, 1.5707963267948966) q[23];
z q[22];
sdg q[15];
y q[2];
rz(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[14];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[7];
