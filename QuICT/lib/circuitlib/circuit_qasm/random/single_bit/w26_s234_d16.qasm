OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
creg c[26];
u3(0, 0, 1.5707963267948966) q[25];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[19];
x q[6];
h q[18];
tdg q[17];
z q[8];
s q[18];
u1(1.5707963267948966) q[3];
s q[14];
sdg q[14];
t q[24];
u3(0, 0, 1.5707963267948966) q[0];
y q[18];
t q[15];
sdg q[1];
u1(1.5707963267948966) q[13];
x q[7];
h q[10];
sdg q[18];
tdg q[25];
z q[24];
x q[4];
ry(1.5707963267948966) q[18];
u1(1.5707963267948966) q[16];
x q[2];
y q[23];
rz(1.5707963267948966) q[17];
h q[3];
rz(1.5707963267948966) q[23];
s q[13];
s q[22];
rz(1.5707963267948966) q[19];
h q[9];
u3(0, 0, 1.5707963267948966) q[22];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[15];
u1(1.5707963267948966) q[12];
y q[18];
sdg q[10];
rz(1.5707963267948966) q[18];
u3(0, 0, 1.5707963267948966) q[20];
sdg q[16];
y q[5];
s q[19];
y q[14];
x q[23];
rx(1.5707963267948966) q[6];
s q[0];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[1];
tdg q[7];
ry(1.5707963267948966) q[6];
z q[10];
sdg q[4];
x q[0];
h q[13];
rx(1.5707963267948966) q[5];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[13];
z q[3];
ry(1.5707963267948966) q[7];
tdg q[3];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[10];
z q[0];
u3(0, 0, 1.5707963267948966) q[23];
s q[4];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[11];
x q[2];
s q[14];
x q[4];
z q[11];
z q[13];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[1];
tdg q[14];
h q[13];
x q[0];
y q[8];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[17];
sdg q[3];
t q[20];
x q[15];
h q[15];
t q[8];
y q[7];
z q[22];
tdg q[17];
x q[24];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[15];
z q[22];
rz(1.5707963267948966) q[13];
y q[15];
rx(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u1(1.5707963267948966) q[18];
tdg q[7];
x q[18];
ry(1.5707963267948966) q[6];
x q[12];
y q[4];
h q[21];
ry(1.5707963267948966) q[8];
t q[14];
ry(1.5707963267948966) q[14];
h q[10];
t q[22];
rz(1.5707963267948966) q[21];
y q[7];
h q[10];
x q[8];
x q[24];
s q[0];
s q[9];
h q[4];
tdg q[16];
s q[12];
ry(1.5707963267948966) q[8];
s q[8];
y q[0];
tdg q[17];
s q[13];
z q[4];
u1(1.5707963267948966) q[6];
u1(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[19];
tdg q[8];
x q[19];
u3(0, 0, 1.5707963267948966) q[14];
tdg q[11];
y q[24];
tdg q[19];
u1(1.5707963267948966) q[2];
y q[9];
t q[3];
ry(1.5707963267948966) q[13];
s q[5];
u2(1.5707963267948966, 1.5707963267948966) q[14];
tdg q[7];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[18];
x q[17];
tdg q[8];
z q[12];
u3(0, 0, 1.5707963267948966) q[2];
x q[7];
h q[14];
sdg q[1];
rx(1.5707963267948966) q[1];
tdg q[10];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[11];
h q[0];
h q[4];
s q[4];
u3(0, 0, 1.5707963267948966) q[17];
sdg q[15];
h q[1];
u1(1.5707963267948966) q[12];
sdg q[3];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[15];
rx(1.5707963267948966) q[17];
x q[16];
u3(0, 0, 1.5707963267948966) q[11];
tdg q[1];
sdg q[23];
h q[5];
sdg q[2];
t q[17];
u2(1.5707963267948966, 1.5707963267948966) q[25];
t q[0];
h q[18];
u3(0, 0, 1.5707963267948966) q[5];
y q[7];
s q[7];
h q[10];
ry(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[16];
rz(1.5707963267948966) q[19];
s q[21];
ry(1.5707963267948966) q[1];
z q[18];
x q[14];
u1(1.5707963267948966) q[2];
x q[1];
u1(1.5707963267948966) q[0];
z q[24];
u3(0, 0, 1.5707963267948966) q[4];
x q[12];
t q[21];
rz(1.5707963267948966) q[14];
z q[6];
t q[18];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[7];
t q[5];
h q[6];
h q[17];
y q[16];
sdg q[3];
z q[0];
u1(1.5707963267948966) q[4];
sdg q[2];
h q[9];
sdg q[13];
u2(1.5707963267948966, 1.5707963267948966) q[20];
rx(1.5707963267948966) q[16];
z q[15];
u2(1.5707963267948966, 1.5707963267948966) q[12];
y q[25];
sdg q[4];
h q[24];
h q[6];
t q[23];
sdg q[6];
z q[3];
ry(1.5707963267948966) q[6];
y q[16];
rz(1.5707963267948966) q[15];
y q[24];
y q[23];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[16];
u1(1.5707963267948966) q[23];
t q[6];