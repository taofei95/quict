OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
creg c[24];
t q[4];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[2];
tdg q[21];
t q[14];
s q[22];
h q[6];
s q[17];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[16];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[17];
sdg q[6];
t q[3];
s q[21];
u1(1.5707963267948966) q[11];
t q[20];
u1(1.5707963267948966) q[22];
ry(1.5707963267948966) q[9];
h q[11];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
x q[22];
ry(1.5707963267948966) q[16];
t q[22];
rz(1.5707963267948966) q[9];
y q[17];
tdg q[17];
h q[2];
u2(1.5707963267948966, 1.5707963267948966) q[22];
rz(1.5707963267948966) q[17];
u1(1.5707963267948966) q[6];
s q[16];
t q[17];
rz(1.5707963267948966) q[15];
h q[21];
t q[8];
tdg q[1];
t q[1];
sdg q[22];
y q[2];
t q[18];
ry(1.5707963267948966) q[9];
s q[17];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[4];
u1(1.5707963267948966) q[23];
s q[4];
sdg q[12];
z q[0];
y q[7];
z q[22];
t q[20];
x q[0];
y q[3];
z q[23];
z q[13];
z q[20];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[11];
z q[18];
t q[0];
z q[2];
s q[18];
y q[17];
tdg q[20];
tdg q[7];
t q[9];
s q[18];
x q[16];
h q[23];
tdg q[0];
rx(1.5707963267948966) q[5];
t q[1];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[9];
y q[17];
tdg q[19];
sdg q[22];
s q[10];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[13];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[18];
s q[3];
y q[5];
ry(1.5707963267948966) q[2];
z q[19];
u3(0, 0, 1.5707963267948966) q[0];
z q[8];
ry(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[5];
h q[8];
u3(0, 0, 1.5707963267948966) q[23];
u2(1.5707963267948966, 1.5707963267948966) q[17];
t q[23];
u3(0, 0, 1.5707963267948966) q[15];
h q[23];
sdg q[19];
u3(0, 0, 1.5707963267948966) q[21];
t q[9];
x q[17];
y q[19];
u3(0, 0, 1.5707963267948966) q[16];
h q[20];
sdg q[12];
u1(1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[13];
y q[19];
x q[13];
z q[1];
tdg q[13];
u1(1.5707963267948966) q[10];
t q[8];
ry(1.5707963267948966) q[2];
x q[1];
tdg q[10];
u1(1.5707963267948966) q[10];
z q[9];
u3(0, 0, 1.5707963267948966) q[7];
y q[23];
h q[14];
rx(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[9];
sdg q[8];
ry(1.5707963267948966) q[10];
s q[21];
s q[10];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
tdg q[0];
t q[3];
t q[10];
h q[17];
x q[8];
rx(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[23];
t q[17];
z q[11];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[21];
x q[7];
tdg q[3];
y q[8];
x q[15];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[14];
u1(1.5707963267948966) q[19];
x q[9];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[23];
s q[5];
sdg q[8];
x q[19];
u1(1.5707963267948966) q[0];
t q[5];
u3(0, 0, 1.5707963267948966) q[22];
u3(0, 0, 1.5707963267948966) q[21];
tdg q[9];
tdg q[9];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[15];
x q[1];
rz(1.5707963267948966) q[20];
