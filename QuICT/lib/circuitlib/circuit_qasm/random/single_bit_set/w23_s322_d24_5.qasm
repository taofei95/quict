OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
u2(1.5707963267948966, 1.5707963267948966) q[10];
rx(1.5707963267948966) q[3];
tdg q[15];
s q[9];
x q[18];
y q[21];
h q[18];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[16];
rz(1.5707963267948966) q[1];
tdg q[11];
rx(1.5707963267948966) q[0];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[12];
y q[21];
z q[9];
y q[9];
t q[0];
rz(1.5707963267948966) q[8];
z q[17];
rx(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[18];
t q[13];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[4];
z q[17];
u2(1.5707963267948966, 1.5707963267948966) q[22];
u3(0, 0, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[22];
tdg q[0];
y q[8];
ry(1.5707963267948966) q[21];
t q[19];
u1(1.5707963267948966) q[16];
s q[21];
u3(0, 0, 1.5707963267948966) q[17];
rx(1.5707963267948966) q[17];
x q[2];
z q[4];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[21];
x q[7];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[17];
tdg q[22];
z q[22];
tdg q[10];
rx(1.5707963267948966) q[21];
s q[2];
s q[7];
u3(0, 0, 1.5707963267948966) q[5];
y q[2];
tdg q[14];
z q[7];
u1(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[8];
z q[0];
h q[15];
z q[19];
rz(1.5707963267948966) q[5];
t q[18];
rx(1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[6];
y q[7];
x q[19];
sdg q[4];
x q[18];
s q[9];
u3(0, 0, 1.5707963267948966) q[9];
rx(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[13];
s q[21];
y q[17];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[22];
u1(1.5707963267948966) q[5];
y q[13];
x q[6];
sdg q[11];
ry(1.5707963267948966) q[9];
h q[9];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[22];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[20];
x q[17];
h q[21];
rz(1.5707963267948966) q[17];
u1(1.5707963267948966) q[0];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[13];
h q[6];
rz(1.5707963267948966) q[0];
sdg q[5];
u1(1.5707963267948966) q[13];
ry(1.5707963267948966) q[18];
s q[11];
rz(1.5707963267948966) q[20];
s q[6];
tdg q[14];
x q[14];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[7];
h q[14];
h q[6];
u1(1.5707963267948966) q[13];
s q[17];
t q[7];
rx(1.5707963267948966) q[20];
tdg q[7];
u1(1.5707963267948966) q[12];
t q[3];
ry(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[22];
u3(0, 0, 1.5707963267948966) q[9];
sdg q[14];
ry(1.5707963267948966) q[19];
h q[4];
s q[21];
t q[14];
s q[19];
sdg q[5];
x q[6];
x q[18];
rz(1.5707963267948966) q[11];
t q[16];
t q[10];
ry(1.5707963267948966) q[10];
h q[6];
x q[13];
u3(0, 0, 1.5707963267948966) q[10];
y q[8];
u3(0, 0, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[15];
s q[0];
tdg q[16];
y q[18];
rx(1.5707963267948966) q[16];
y q[12];
h q[8];
u1(1.5707963267948966) q[7];
t q[10];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[3];
y q[0];
u3(0, 0, 1.5707963267948966) q[17];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[4];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[17];
y q[22];
s q[19];
tdg q[10];
sdg q[10];
h q[7];
x q[13];
s q[3];
sdg q[19];
ry(1.5707963267948966) q[13];
t q[18];
tdg q[9];
y q[12];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[8];
s q[13];
ry(1.5707963267948966) q[6];
u1(1.5707963267948966) q[8];
z q[17];
t q[16];
x q[6];
u3(0, 0, 1.5707963267948966) q[19];
y q[8];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u1(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[6];
t q[15];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[18];
u1(1.5707963267948966) q[16];
z q[10];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[10];
tdg q[6];
u3(0, 0, 1.5707963267948966) q[15];
y q[0];
z q[15];
u1(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[10];
rz(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[9];
t q[13];
t q[0];
t q[11];
x q[22];
s q[6];
u2(1.5707963267948966, 1.5707963267948966) q[22];
s q[3];
u3(0, 0, 1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[18];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[17];
x q[6];
ry(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[1];
y q[18];
sdg q[13];
y q[22];
z q[20];
t q[10];
h q[17];
sdg q[7];
u2(1.5707963267948966, 1.5707963267948966) q[22];
tdg q[6];
x q[2];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[14];
t q[17];
y q[17];
rx(1.5707963267948966) q[7];
s q[1];
tdg q[19];
t q[5];
y q[3];
tdg q[11];
y q[8];
t q[17];
z q[20];
u3(0, 0, 1.5707963267948966) q[20];
rx(1.5707963267948966) q[3];
s q[12];
u2(1.5707963267948966, 1.5707963267948966) q[16];
tdg q[10];
t q[7];
sdg q[22];
z q[1];
sdg q[8];
h q[6];
s q[13];
t q[17];
rx(1.5707963267948966) q[10];
tdg q[8];
t q[15];
t q[4];
z q[20];
z q[12];
t q[0];
s q[20];
u2(1.5707963267948966, 1.5707963267948966) q[20];
y q[1];
t q[9];
ry(1.5707963267948966) q[19];
s q[3];
t q[11];
t q[18];
y q[13];
u1(1.5707963267948966) q[16];
h q[8];
ry(1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[18];
rx(1.5707963267948966) q[22];
ry(1.5707963267948966) q[17];
h q[21];
t q[13];
u1(1.5707963267948966) q[5];
sdg q[9];
z q[13];
x q[4];
u3(0, 0, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[9];
s q[15];
u3(0, 0, 1.5707963267948966) q[2];
t q[11];
u2(1.5707963267948966, 1.5707963267948966) q[17];
h q[17];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[17];
y q[20];
sdg q[14];
rx(1.5707963267948966) q[20];
u1(1.5707963267948966) q[8];
h q[9];
sdg q[22];
y q[16];
tdg q[18];
s q[14];
sdg q[9];
x q[1];
sdg q[22];
u1(1.5707963267948966) q[7];
y q[17];
y q[20];
u1(1.5707963267948966) q[22];
sdg q[1];
tdg q[12];
h q[15];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[11];
t q[15];
s q[5];
sdg q[10];
h q[16];
rz(1.5707963267948966) q[4];
tdg q[19];
z q[12];
u1(1.5707963267948966) q[17];
sdg q[9];
