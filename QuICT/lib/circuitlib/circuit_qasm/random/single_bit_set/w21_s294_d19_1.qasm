OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
rx(1.5707963267948966) q[14];
x q[12];
rx(1.5707963267948966) q[16];
u1(1.5707963267948966) q[4];
s q[15];
u3(0, 0, 1.5707963267948966) q[6];
rx(1.5707963267948966) q[14];
u1(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[4];
s q[4];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[5];
y q[9];
y q[3];
h q[8];
h q[12];
ry(1.5707963267948966) q[5];
s q[16];
rx(1.5707963267948966) q[10];
sdg q[0];
z q[3];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[14];
s q[19];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[6];
z q[14];
z q[17];
t q[9];
ry(1.5707963267948966) q[20];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[9];
x q[18];
rx(1.5707963267948966) q[18];
x q[7];
z q[8];
z q[8];
t q[19];
y q[14];
u1(1.5707963267948966) q[3];
t q[17];
u1(1.5707963267948966) q[6];
t q[9];
rx(1.5707963267948966) q[8];
t q[6];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[4];
y q[7];
u1(1.5707963267948966) q[17];
x q[1];
x q[9];
sdg q[13];
u3(0, 0, 1.5707963267948966) q[10];
h q[0];
x q[1];
s q[8];
y q[20];
y q[12];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[2];
sdg q[7];
tdg q[19];
h q[12];
y q[10];
s q[10];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[10];
h q[13];
u2(1.5707963267948966, 1.5707963267948966) q[18];
sdg q[9];
u3(0, 0, 1.5707963267948966) q[18];
h q[8];
z q[4];
rx(1.5707963267948966) q[7];
s q[7];
u1(1.5707963267948966) q[6];
y q[13];
tdg q[4];
rz(1.5707963267948966) q[13];
x q[17];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[19];
ry(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[6];
h q[15];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[17];
z q[0];
ry(1.5707963267948966) q[9];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[7];
h q[19];
s q[19];
u1(1.5707963267948966) q[8];
sdg q[3];
h q[13];
ry(1.5707963267948966) q[20];
h q[0];
t q[18];
h q[9];
t q[19];
t q[6];
tdg q[9];
ry(1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[11];
x q[12];
y q[18];
ry(1.5707963267948966) q[14];
t q[3];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[12];
u1(1.5707963267948966) q[2];
tdg q[18];
h q[16];
s q[8];
y q[19];
s q[17];
tdg q[11];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[13];
tdg q[17];
ry(1.5707963267948966) q[10];
y q[14];
u1(1.5707963267948966) q[7];
h q[14];
t q[8];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[3];
sdg q[10];
ry(1.5707963267948966) q[2];
z q[4];
sdg q[3];
t q[6];
h q[10];
tdg q[18];
x q[5];
z q[11];
y q[19];
ry(1.5707963267948966) q[2];
s q[2];
tdg q[19];
u1(1.5707963267948966) q[5];
z q[6];
t q[11];
u1(1.5707963267948966) q[16];
sdg q[13];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[17];
sdg q[8];
h q[19];
rx(1.5707963267948966) q[19];
u1(1.5707963267948966) q[13];
x q[0];
rz(1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[4];
s q[5];
t q[0];
rz(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[7];
rz(1.5707963267948966) q[11];
s q[6];
sdg q[3];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[19];
s q[3];
h q[20];
sdg q[6];
s q[1];
y q[11];
t q[1];
ry(1.5707963267948966) q[13];
x q[11];
y q[6];
tdg q[7];
tdg q[1];
s q[11];
x q[16];
x q[16];
h q[8];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[19];
u1(1.5707963267948966) q[8];
x q[15];
s q[12];
tdg q[18];
x q[1];
ry(1.5707963267948966) q[15];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[1];
s q[6];
u1(1.5707963267948966) q[16];
x q[19];
z q[9];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[12];
u1(1.5707963267948966) q[17];
x q[0];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[16];
rz(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[9];
s q[3];
sdg q[5];
tdg q[14];
s q[19];
u3(0, 0, 1.5707963267948966) q[20];
t q[3];
x q[11];
rx(1.5707963267948966) q[12];
h q[14];
tdg q[13];
h q[19];
y q[3];
z q[7];
z q[5];
x q[8];
sdg q[12];
u3(0, 0, 1.5707963267948966) q[11];
x q[17];
rz(1.5707963267948966) q[20];
s q[7];
z q[15];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[6];
y q[20];
ry(1.5707963267948966) q[4];
h q[18];
u3(0, 0, 1.5707963267948966) q[14];
sdg q[13];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[10];
t q[18];
x q[15];
u1(1.5707963267948966) q[18];
rx(1.5707963267948966) q[13];
sdg q[2];
rz(1.5707963267948966) q[17];
u1(1.5707963267948966) q[6];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[14];
ry(1.5707963267948966) q[11];
sdg q[17];
z q[17];
sdg q[0];
x q[20];
u1(1.5707963267948966) q[8];
y q[17];
t q[3];
ry(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[0];
z q[11];
sdg q[7];
y q[14];
s q[4];
tdg q[2];
sdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[14];
h q[20];
ry(1.5707963267948966) q[9];
h q[13];
h q[12];
tdg q[2];
u3(0, 0, 1.5707963267948966) q[20];
ry(1.5707963267948966) q[9];
s q[18];
y q[8];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
y q[0];
ry(1.5707963267948966) q[12];
t q[10];
x q[2];
s q[19];
sdg q[1];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[0];
t q[14];
z q[13];
z q[20];
h q[17];
rz(1.5707963267948966) q[20];
tdg q[12];
