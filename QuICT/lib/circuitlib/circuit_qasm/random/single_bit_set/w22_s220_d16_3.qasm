OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
ry(1.5707963267948966) q[10];
tdg q[3];
z q[3];
sdg q[0];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[9];
ry(1.5707963267948966) q[0];
y q[14];
ry(1.5707963267948966) q[10];
sdg q[17];
z q[14];
u1(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[15];
tdg q[8];
s q[10];
y q[4];
h q[12];
u1(1.5707963267948966) q[9];
s q[3];
t q[15];
ry(1.5707963267948966) q[19];
h q[1];
rz(1.5707963267948966) q[13];
tdg q[2];
t q[14];
h q[5];
rx(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[18];
s q[11];
ry(1.5707963267948966) q[5];
x q[20];
u1(1.5707963267948966) q[6];
x q[20];
y q[5];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
y q[20];
t q[3];
h q[2];
u3(0, 0, 1.5707963267948966) q[4];
t q[20];
tdg q[17];
sdg q[10];
h q[16];
u2(1.5707963267948966, 1.5707963267948966) q[6];
t q[0];
h q[19];
u3(0, 0, 1.5707963267948966) q[15];
rz(1.5707963267948966) q[3];
tdg q[13];
u3(0, 0, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[11];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[10];
rx(1.5707963267948966) q[7];
sdg q[14];
z q[13];
y q[6];
z q[4];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[3];
t q[3];
y q[6];
u3(0, 0, 1.5707963267948966) q[2];
t q[6];
ry(1.5707963267948966) q[10];
u1(1.5707963267948966) q[8];
h q[5];
t q[0];
tdg q[14];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[2];
y q[12];
ry(1.5707963267948966) q[16];
x q[11];
tdg q[8];
ry(1.5707963267948966) q[15];
z q[8];
y q[12];
u3(0, 0, 1.5707963267948966) q[1];
sdg q[20];
u2(1.5707963267948966, 1.5707963267948966) q[21];
u1(1.5707963267948966) q[0];
y q[2];
u3(0, 0, 1.5707963267948966) q[14];
y q[14];
t q[14];
s q[0];
tdg q[5];
h q[16];
ry(1.5707963267948966) q[10];
sdg q[4];
t q[9];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[13];
s q[5];
h q[13];
h q[9];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[21];
t q[9];
z q[21];
x q[13];
sdg q[16];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[14];
s q[9];
s q[10];
rx(1.5707963267948966) q[14];
u3(0, 0, 1.5707963267948966) q[13];
s q[3];
u3(0, 0, 1.5707963267948966) q[15];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[13];
u1(1.5707963267948966) q[10];
rx(1.5707963267948966) q[0];
y q[10];
sdg q[3];
t q[7];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[10];
u3(0, 0, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[3];
x q[13];
rz(1.5707963267948966) q[14];
h q[16];
z q[9];
u2(1.5707963267948966, 1.5707963267948966) q[1];
s q[5];
rx(1.5707963267948966) q[9];
sdg q[0];
y q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[19];
s q[18];
z q[3];
u2(1.5707963267948966, 1.5707963267948966) q[20];
y q[14];
h q[14];
u3(0, 0, 1.5707963267948966) q[19];
y q[4];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[12];
s q[8];
s q[15];
tdg q[6];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[13];
u1(1.5707963267948966) q[10];
rx(1.5707963267948966) q[21];
u1(1.5707963267948966) q[20];
z q[6];
ry(1.5707963267948966) q[4];
z q[12];
h q[6];
sdg q[1];
t q[11];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[17];
u1(1.5707963267948966) q[4];
rz(1.5707963267948966) q[20];
ry(1.5707963267948966) q[9];
y q[8];
z q[19];
rz(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[16];
y q[1];
u1(1.5707963267948966) q[21];
y q[16];
h q[16];
x q[7];
sdg q[19];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[1];
y q[8];
x q[6];
ry(1.5707963267948966) q[9];
x q[5];
ry(1.5707963267948966) q[11];
h q[11];
x q[1];
u1(1.5707963267948966) q[4];
y q[9];
h q[5];
t q[16];
u3(0, 0, 1.5707963267948966) q[16];
t q[10];
t q[17];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[12];
h q[5];
y q[19];
u3(0, 0, 1.5707963267948966) q[20];
rz(1.5707963267948966) q[10];
s q[13];
z q[21];
rx(1.5707963267948966) q[1];
y q[21];
t q[13];
t q[15];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[15];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[6];
y q[15];
