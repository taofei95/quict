OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
u1(1.5707963267948966) q[16];
s q[10];
t q[5];
rx(1.5707963267948966) q[20];
s q[2];
u1(1.5707963267948966) q[10];
s q[15];
s q[15];
u1(1.5707963267948966) q[13];
ry(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[8];
h q[1];
rz(1.5707963267948966) q[1];
h q[11];
tdg q[2];
ry(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[10];
s q[0];
u1(1.5707963267948966) q[6];
h q[17];
u3(0, 0, 1.5707963267948966) q[8];
sdg q[20];
sdg q[16];
u3(0, 0, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[21];
tdg q[4];
t q[17];
rx(1.5707963267948966) q[11];
u1(1.5707963267948966) q[13];
s q[17];
z q[2];
rz(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
sdg q[9];
x q[9];
u3(0, 0, 1.5707963267948966) q[1];
z q[0];
ry(1.5707963267948966) q[2];
sdg q[8];
s q[0];
y q[3];
tdg q[6];
tdg q[0];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[13];
z q[15];
u2(1.5707963267948966, 1.5707963267948966) q[10];
y q[11];
sdg q[11];
sdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[5];
z q[19];
y q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
x q[16];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[5];
z q[12];
x q[9];
z q[3];
t q[15];
u1(1.5707963267948966) q[11];
y q[14];
u1(1.5707963267948966) q[0];
tdg q[11];
t q[8];
s q[20];
u3(0, 0, 1.5707963267948966) q[7];
tdg q[6];
u1(1.5707963267948966) q[19];
ry(1.5707963267948966) q[2];
z q[9];
tdg q[4];
t q[12];
rz(1.5707963267948966) q[8];
x q[17];
rz(1.5707963267948966) q[20];
x q[10];
u1(1.5707963267948966) q[8];
z q[2];
x q[9];
t q[6];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[2];
s q[18];
tdg q[14];
h q[21];
tdg q[3];
rx(1.5707963267948966) q[3];
z q[17];
u3(0, 0, 1.5707963267948966) q[2];
t q[18];
u2(1.5707963267948966, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[16];
y q[10];
h q[5];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[19];
tdg q[3];
x q[4];
sdg q[3];
sdg q[12];
x q[14];
t q[19];
u1(1.5707963267948966) q[19];
sdg q[15];
rx(1.5707963267948966) q[1];
tdg q[13];
x q[18];
h q[15];
z q[5];
tdg q[14];
rx(1.5707963267948966) q[21];
rz(1.5707963267948966) q[21];
u1(1.5707963267948966) q[10];
x q[21];
u2(1.5707963267948966, 1.5707963267948966) q[21];
x q[2];
t q[3];
h q[15];
sdg q[21];
tdg q[1];
t q[18];
rz(1.5707963267948966) q[9];
z q[17];
s q[20];
sdg q[13];
sdg q[13];
rz(1.5707963267948966) q[12];
sdg q[10];
ry(1.5707963267948966) q[6];
s q[18];
rx(1.5707963267948966) q[8];
s q[3];
x q[3];
u3(0, 0, 1.5707963267948966) q[11];
z q[4];
rz(1.5707963267948966) q[7];
y q[19];
h q[10];
x q[2];
sdg q[19];
rx(1.5707963267948966) q[3];
sdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[13];
x q[13];
sdg q[17];
u2(1.5707963267948966, 1.5707963267948966) q[16];
ry(1.5707963267948966) q[6];
u1(1.5707963267948966) q[0];
z q[14];
y q[12];
y q[16];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rx(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[21];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[15];
h q[3];
ry(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[20];
h q[18];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[12];
x q[3];
s q[9];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[16];
y q[7];
tdg q[2];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[12];
