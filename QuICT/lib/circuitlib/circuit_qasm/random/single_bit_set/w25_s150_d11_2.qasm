OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
x q[12];
z q[0];
tdg q[4];
u1(1.5707963267948966) q[22];
x q[11];
x q[17];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[1];
rz(1.5707963267948966) q[15];
x q[18];
y q[18];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[18];
tdg q[11];
x q[11];
y q[13];
u1(1.5707963267948966) q[7];
x q[24];
t q[15];
u1(1.5707963267948966) q[4];
ry(1.5707963267948966) q[23];
tdg q[24];
ry(1.5707963267948966) q[20];
x q[18];
t q[2];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[7];
t q[9];
y q[14];
u3(0, 0, 1.5707963267948966) q[21];
rx(1.5707963267948966) q[18];
x q[20];
ry(1.5707963267948966) q[18];
z q[22];
sdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[21];
s q[13];
sdg q[19];
u1(1.5707963267948966) q[15];
s q[4];
h q[16];
rx(1.5707963267948966) q[6];
tdg q[24];
tdg q[24];
u1(1.5707963267948966) q[10];
t q[16];
ry(1.5707963267948966) q[23];
z q[11];
rx(1.5707963267948966) q[24];
z q[16];
h q[12];
rx(1.5707963267948966) q[9];
y q[15];
ry(1.5707963267948966) q[16];
t q[3];
t q[15];
ry(1.5707963267948966) q[20];
s q[7];
x q[20];
x q[18];
y q[16];
x q[12];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[23];
rx(1.5707963267948966) q[4];
z q[14];
ry(1.5707963267948966) q[11];
x q[13];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[3];
s q[14];
s q[15];
tdg q[5];
x q[16];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[13];
ry(1.5707963267948966) q[20];
rz(1.5707963267948966) q[24];
tdg q[1];
y q[16];
s q[2];
u3(0, 0, 1.5707963267948966) q[17];
x q[1];
u3(0, 0, 1.5707963267948966) q[21];
y q[3];
u3(0, 0, 1.5707963267948966) q[19];
rx(1.5707963267948966) q[7];
tdg q[5];
t q[11];
ry(1.5707963267948966) q[7];
t q[14];
h q[8];
s q[3];
ry(1.5707963267948966) q[11];
z q[6];
u2(1.5707963267948966, 1.5707963267948966) q[22];
s q[12];
u2(1.5707963267948966, 1.5707963267948966) q[8];
s q[11];
u3(0, 0, 1.5707963267948966) q[15];
y q[23];
s q[17];
rz(1.5707963267948966) q[12];
tdg q[4];
tdg q[3];
x q[16];
s q[7];
tdg q[21];
t q[20];
y q[22];
u3(0, 0, 1.5707963267948966) q[13];
rx(1.5707963267948966) q[1];
s q[21];
rz(1.5707963267948966) q[23];
z q[24];
sdg q[18];
u2(1.5707963267948966, 1.5707963267948966) q[17];
h q[18];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[15];
z q[18];
ry(1.5707963267948966) q[8];
tdg q[21];
x q[20];
ry(1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[22];
y q[10];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[10];
z q[6];
sdg q[1];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
sdg q[10];
u3(0, 0, 1.5707963267948966) q[24];
s q[23];
y q[6];
u2(1.5707963267948966, 1.5707963267948966) q[21];
s q[4];
s q[0];
y q[10];
s q[23];
u2(1.5707963267948966, 1.5707963267948966) q[23];
y q[23];
