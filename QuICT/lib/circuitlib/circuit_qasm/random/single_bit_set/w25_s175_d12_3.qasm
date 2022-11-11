OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
u3(0, 0, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
u1(1.5707963267948966) q[19];
ry(1.5707963267948966) q[5];
z q[11];
u2(1.5707963267948966, 1.5707963267948966) q[19];
sdg q[14];
sdg q[2];
rx(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[14];
t q[21];
z q[12];
z q[13];
t q[16];
u3(0, 0, 1.5707963267948966) q[19];
rz(1.5707963267948966) q[13];
x q[19];
z q[12];
h q[2];
tdg q[10];
h q[23];
u2(1.5707963267948966, 1.5707963267948966) q[18];
t q[12];
rx(1.5707963267948966) q[19];
tdg q[17];
tdg q[12];
x q[10];
u2(1.5707963267948966, 1.5707963267948966) q[9];
z q[20];
t q[14];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[4];
z q[20];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[23];
u2(1.5707963267948966, 1.5707963267948966) q[8];
y q[4];
h q[17];
u3(0, 0, 1.5707963267948966) q[12];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[0];
sdg q[22];
y q[1];
z q[24];
x q[23];
y q[19];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[21];
rx(1.5707963267948966) q[16];
sdg q[22];
t q[16];
u1(1.5707963267948966) q[12];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[0];
s q[18];
ry(1.5707963267948966) q[10];
z q[3];
t q[11];
rx(1.5707963267948966) q[20];
u1(1.5707963267948966) q[0];
s q[11];
t q[2];
x q[9];
x q[7];
h q[23];
u1(1.5707963267948966) q[14];
t q[4];
s q[18];
h q[17];
x q[6];
t q[14];
y q[19];
s q[8];
x q[19];
ry(1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[22];
rz(1.5707963267948966) q[24];
h q[12];
h q[11];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[0];
tdg q[3];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[8];
y q[8];
u2(1.5707963267948966, 1.5707963267948966) q[5];
z q[20];
h q[9];
h q[18];
u2(1.5707963267948966, 1.5707963267948966) q[23];
u1(1.5707963267948966) q[1];
y q[11];
s q[8];
t q[1];
t q[3];
y q[15];
u3(0, 0, 1.5707963267948966) q[21];
u2(1.5707963267948966, 1.5707963267948966) q[15];
tdg q[9];
u1(1.5707963267948966) q[16];
ry(1.5707963267948966) q[24];
y q[19];
sdg q[20];
u1(1.5707963267948966) q[13];
y q[14];
h q[16];
s q[8];
t q[1];
x q[8];
x q[1];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[16];
sdg q[16];
t q[11];
y q[11];
u3(0, 0, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[5];
z q[5];
sdg q[23];
h q[0];
ry(1.5707963267948966) q[24];
u1(1.5707963267948966) q[13];
rx(1.5707963267948966) q[23];
y q[15];
t q[8];
tdg q[11];
rz(1.5707963267948966) q[17];
ry(1.5707963267948966) q[8];
h q[11];
rx(1.5707963267948966) q[2];
t q[24];
t q[10];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[7];
tdg q[23];
sdg q[24];
u1(1.5707963267948966) q[9];
h q[2];
tdg q[6];
y q[13];
t q[5];
u1(1.5707963267948966) q[16];
t q[12];
y q[16];
u3(0, 0, 1.5707963267948966) q[10];
h q[6];
x q[5];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[15];
s q[20];
z q[19];
rx(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[9];
t q[5];
u1(1.5707963267948966) q[0];
sdg q[7];
rz(1.5707963267948966) q[17];
z q[16];
s q[7];
rz(1.5707963267948966) q[10];
sdg q[0];
z q[19];
rx(1.5707963267948966) q[4];
t q[5];
rx(1.5707963267948966) q[15];
z q[11];
t q[21];
x q[19];
ry(1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[15];
