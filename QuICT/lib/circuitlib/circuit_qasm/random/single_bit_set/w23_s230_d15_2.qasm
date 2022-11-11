OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
ry(1.5707963267948966) q[18];
x q[7];
s q[18];
x q[12];
u2(1.5707963267948966, 1.5707963267948966) q[1];
s q[18];
h q[6];
z q[7];
rz(1.5707963267948966) q[17];
ry(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[13];
u1(1.5707963267948966) q[2];
z q[11];
z q[12];
sdg q[12];
u3(0, 0, 1.5707963267948966) q[16];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
h q[1];
s q[15];
u1(1.5707963267948966) q[3];
sdg q[13];
sdg q[18];
t q[12];
u3(0, 0, 1.5707963267948966) q[17];
z q[10];
s q[1];
rz(1.5707963267948966) q[5];
h q[11];
rz(1.5707963267948966) q[4];
y q[19];
x q[17];
s q[3];
t q[15];
t q[9];
h q[10];
tdg q[14];
t q[14];
tdg q[19];
t q[6];
t q[4];
z q[4];
x q[21];
z q[7];
z q[21];
u3(0, 0, 1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[11];
rz(1.5707963267948966) q[14];
t q[4];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[9];
y q[15];
z q[21];
x q[0];
ry(1.5707963267948966) q[1];
s q[6];
z q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[20];
t q[7];
z q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
y q[21];
t q[21];
t q[20];
tdg q[11];
h q[19];
x q[13];
sdg q[7];
u1(1.5707963267948966) q[17];
h q[13];
t q[22];
y q[0];
s q[13];
u3(0, 0, 1.5707963267948966) q[0];
t q[12];
u3(0, 0, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[9];
h q[5];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[20];
u1(1.5707963267948966) q[5];
sdg q[6];
y q[6];
rx(1.5707963267948966) q[14];
z q[4];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[10];
t q[19];
s q[15];
u1(1.5707963267948966) q[13];
x q[15];
rx(1.5707963267948966) q[1];
s q[19];
s q[19];
x q[20];
y q[22];
x q[13];
h q[19];
s q[10];
tdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[20];
z q[16];
x q[21];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[8];
rx(1.5707963267948966) q[19];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[15];
h q[19];
x q[11];
u3(0, 0, 1.5707963267948966) q[18];
tdg q[22];
h q[16];
u3(0, 0, 1.5707963267948966) q[10];
sdg q[12];
z q[0];
sdg q[8];
u1(1.5707963267948966) q[4];
h q[10];
sdg q[12];
h q[13];
u1(1.5707963267948966) q[8];
tdg q[6];
s q[1];
t q[13];
y q[19];
tdg q[6];
x q[10];
y q[21];
tdg q[8];
h q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
h q[6];
rz(1.5707963267948966) q[6];
tdg q[22];
u3(0, 0, 1.5707963267948966) q[5];
s q[9];
x q[19];
x q[2];
ry(1.5707963267948966) q[22];
sdg q[22];
sdg q[13];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[15];
tdg q[22];
u1(1.5707963267948966) q[22];
rz(1.5707963267948966) q[21];
u2(1.5707963267948966, 1.5707963267948966) q[17];
h q[15];
x q[15];
x q[19];
x q[12];
u3(0, 0, 1.5707963267948966) q[10];
s q[9];
h q[0];
s q[12];
t q[18];
tdg q[18];
sdg q[9];
z q[15];
t q[1];
tdg q[0];
z q[9];
t q[16];
ry(1.5707963267948966) q[17];
h q[7];
rz(1.5707963267948966) q[18];
u1(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[8];
h q[11];
y q[11];
rx(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[12];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[7];
rx(1.5707963267948966) q[6];
h q[9];
u2(1.5707963267948966, 1.5707963267948966) q[18];
u3(0, 0, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[17];
h q[16];
t q[3];
h q[21];
rz(1.5707963267948966) q[9];
h q[9];
u3(0, 0, 1.5707963267948966) q[22];
h q[22];
u1(1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
z q[2];
x q[0];
sdg q[21];
x q[0];
x q[17];
t q[21];
t q[21];
t q[21];
x q[3];
u3(0, 0, 1.5707963267948966) q[19];
tdg q[8];
rx(1.5707963267948966) q[21];
t q[18];
x q[9];
u3(0, 0, 1.5707963267948966) q[22];
t q[11];
z q[2];
s q[9];
u1(1.5707963267948966) q[9];
y q[7];
u3(0, 0, 1.5707963267948966) q[22];
tdg q[11];
tdg q[2];
s q[20];
x q[9];
u1(1.5707963267948966) q[18];
rx(1.5707963267948966) q[21];
s q[20];
x q[13];
rz(1.5707963267948966) q[19];
