OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
x q[20];
s q[0];
s q[7];
tdg q[3];
tdg q[11];
x q[0];
sdg q[12];
h q[10];
rx(1.5707963267948966) q[18];
s q[2];
u1(1.5707963267948966) q[11];
rx(1.5707963267948966) q[10];
x q[5];
u2(1.5707963267948966, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[16];
tdg q[17];
rx(1.5707963267948966) q[7];
z q[15];
x q[5];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
s q[21];
t q[11];
ry(1.5707963267948966) q[15];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[4];
y q[12];
u2(1.5707963267948966, 1.5707963267948966) q[10];
z q[11];
s q[16];
u1(1.5707963267948966) q[6];
rz(1.5707963267948966) q[7];
x q[10];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[15];
y q[6];
z q[2];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u1(1.5707963267948966) q[2];
t q[15];
ry(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[4];
h q[14];
h q[6];
rx(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[20];
h q[4];
h q[12];
rz(1.5707963267948966) q[1];
tdg q[5];
s q[11];
ry(1.5707963267948966) q[10];
tdg q[14];
x q[15];
t q[14];
u1(1.5707963267948966) q[7];
tdg q[19];
x q[7];
y q[7];
h q[11];
rx(1.5707963267948966) q[13];
z q[16];
ry(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[14];
s q[20];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[18];
sdg q[14];
tdg q[13];
u3(0, 0, 1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[16];
x q[5];
u2(1.5707963267948966, 1.5707963267948966) q[9];
y q[8];
h q[8];
z q[16];
x q[15];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[12];
tdg q[10];
u2(1.5707963267948966, 1.5707963267948966) q[5];
z q[20];
u1(1.5707963267948966) q[15];
t q[8];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[4];
z q[8];
z q[5];
t q[4];
ry(1.5707963267948966) q[15];
z q[4];
y q[20];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[4];
x q[11];
u3(0, 0, 1.5707963267948966) q[15];
sdg q[13];
x q[15];
s q[8];
ry(1.5707963267948966) q[18];
ry(1.5707963267948966) q[14];
u1(1.5707963267948966) q[15];
tdg q[19];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[8];
sdg q[19];
tdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[15];
t q[4];
tdg q[18];
rx(1.5707963267948966) q[18];
rx(1.5707963267948966) q[17];
z q[4];
sdg q[12];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[10];
sdg q[10];
sdg q[9];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[21];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[12];
s q[7];
sdg q[16];
rx(1.5707963267948966) q[11];
z q[7];
rz(1.5707963267948966) q[0];
tdg q[12];
x q[15];
sdg q[20];
sdg q[6];
sdg q[3];
z q[7];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[11];
x q[17];
tdg q[20];
z q[14];
ry(1.5707963267948966) q[10];
y q[21];
rx(1.5707963267948966) q[11];
tdg q[17];
tdg q[17];
h q[1];
u3(0, 0, 1.5707963267948966) q[7];
z q[13];
t q[8];
h q[15];
z q[17];
t q[17];
ry(1.5707963267948966) q[9];
sdg q[17];
z q[21];
u3(0, 0, 1.5707963267948966) q[0];
z q[5];
x q[9];
tdg q[3];
u1(1.5707963267948966) q[17];
z q[10];
t q[9];
t q[20];
y q[7];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[14];
s q[0];
u3(0, 0, 1.5707963267948966) q[5];
t q[19];
s q[7];
sdg q[0];
s q[8];
x q[1];
u3(0, 0, 1.5707963267948966) q[19];
x q[20];
x q[3];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[14];
x q[4];
rz(1.5707963267948966) q[9];
s q[21];
t q[17];
tdg q[8];
x q[10];
x q[14];
s q[7];
rz(1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[10];
rz(1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[11];
sdg q[0];
y q[3];
h q[21];
t q[6];
ry(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[8];
x q[10];
u2(1.5707963267948966, 1.5707963267948966) q[21];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[20];
rz(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[17];
h q[13];
rz(1.5707963267948966) q[20];
y q[20];
rx(1.5707963267948966) q[4];
sdg q[9];
h q[3];
s q[15];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[10];
z q[7];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[9];