OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
z q[3];
x q[7];
s q[13];
z q[2];
sdg q[13];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u1(1.5707963267948966) q[0];
z q[0];
rz(1.5707963267948966) q[2];
y q[4];
x q[18];
x q[18];
t q[2];
ry(1.5707963267948966) q[16];
y q[20];
u3(0, 0, 1.5707963267948966) q[20];
sdg q[14];
rz(1.5707963267948966) q[11];
y q[16];
rz(1.5707963267948966) q[3];
sdg q[2];
u1(1.5707963267948966) q[2];
t q[22];
x q[12];
s q[7];
tdg q[13];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[17];
h q[8];
u3(0, 0, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
s q[16];
s q[11];
rx(1.5707963267948966) q[12];
x q[5];
x q[22];
y q[13];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[14];
tdg q[12];
rz(1.5707963267948966) q[5];
s q[4];
ry(1.5707963267948966) q[5];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[14];
ry(1.5707963267948966) q[21];
t q[13];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[3];
y q[5];
rz(1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[14];
z q[18];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[20];
tdg q[18];
tdg q[2];
tdg q[2];
y q[0];
t q[13];
rx(1.5707963267948966) q[14];
t q[10];
x q[0];
h q[0];
y q[18];
h q[2];
z q[7];
z q[16];
x q[9];
z q[8];
u3(0, 0, 1.5707963267948966) q[20];
u2(1.5707963267948966, 1.5707963267948966) q[20];
s q[20];
ry(1.5707963267948966) q[17];
h q[15];
h q[13];
h q[11];
sdg q[9];
ry(1.5707963267948966) q[11];
sdg q[5];
z q[16];
z q[19];
u3(0, 0, 1.5707963267948966) q[14];
z q[21];
u3(0, 0, 1.5707963267948966) q[20];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[2];
y q[9];
x q[10];
x q[21];
y q[15];
u1(1.5707963267948966) q[11];
tdg q[4];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[21];
sdg q[16];
rz(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[17];
t q[12];
tdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[14];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[8];
sdg q[21];
tdg q[14];
s q[8];
tdg q[6];
t q[11];
s q[15];
t q[6];
u3(0, 0, 1.5707963267948966) q[6];
t q[12];
z q[3];
u3(0, 0, 1.5707963267948966) q[15];
y q[9];
u1(1.5707963267948966) q[14];
y q[18];
x q[6];
u3(0, 0, 1.5707963267948966) q[20];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[19];
rx(1.5707963267948966) q[3];
t q[22];
x q[1];
ry(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[11];
tdg q[21];
ry(1.5707963267948966) q[11];
x q[1];
s q[15];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[13];
t q[5];
y q[10];
x q[6];
z q[13];
t q[1];
tdg q[5];
sdg q[18];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[14];
t q[18];
rz(1.5707963267948966) q[0];
h q[2];
x q[11];
rx(1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[21];
u1(1.5707963267948966) q[11];
z q[4];
ry(1.5707963267948966) q[20];
tdg q[17];
ry(1.5707963267948966) q[8];
t q[12];
rz(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[20];
x q[5];
u2(1.5707963267948966, 1.5707963267948966) q[22];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[2];
y q[14];
ry(1.5707963267948966) q[21];
rz(1.5707963267948966) q[13];
s q[8];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[12];
u1(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[18];
u3(0, 0, 1.5707963267948966) q[13];
y q[17];
tdg q[10];
s q[4];
t q[21];
rz(1.5707963267948966) q[9];
s q[2];
s q[9];
u2(1.5707963267948966, 1.5707963267948966) q[22];
u1(1.5707963267948966) q[21];
rx(1.5707963267948966) q[16];
y q[12];
rx(1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
z q[13];
y q[6];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[11];
u1(1.5707963267948966) q[7];
rx(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[19];
s q[9];
x q[16];
y q[12];
x q[7];
z q[3];
u3(0, 0, 1.5707963267948966) q[1];