OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
sdg q[7];
sdg q[21];
ry(1.5707963267948966) q[7];
tdg q[5];
tdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[9];
y q[19];
u1(1.5707963267948966) q[3];
z q[6];
h q[16];
u3(0, 0, 1.5707963267948966) q[14];
rz(1.5707963267948966) q[0];
s q[2];
h q[16];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[1];
y q[3];
y q[15];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[19];
z q[11];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[14];
u1(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[4];
t q[12];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[17];
s q[20];
sdg q[20];
sdg q[11];
t q[5];
u3(0, 0, 1.5707963267948966) q[11];
t q[0];
u1(1.5707963267948966) q[9];
z q[9];
z q[18];
h q[10];
y q[19];
s q[0];
h q[11];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[11];
x q[13];
sdg q[14];
u1(1.5707963267948966) q[1];
sdg q[3];
z q[20];
sdg q[7];
x q[0];
ry(1.5707963267948966) q[20];
sdg q[9];
x q[7];
s q[3];
t q[8];
s q[8];
z q[4];
ry(1.5707963267948966) q[13];
h q[4];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[3];
sdg q[3];
z q[8];
ry(1.5707963267948966) q[18];
ry(1.5707963267948966) q[21];
y q[17];
x q[17];
sdg q[9];
t q[0];
y q[12];
x q[6];
u2(1.5707963267948966, 1.5707963267948966) q[16];
y q[4];
z q[15];
sdg q[21];
t q[7];
u1(1.5707963267948966) q[18];
y q[19];
u1(1.5707963267948966) q[12];
u1(1.5707963267948966) q[19];
y q[10];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[15];
x q[9];
u1(1.5707963267948966) q[11];
t q[2];
u1(1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[8];
sdg q[9];
sdg q[13];
y q[1];
u1(1.5707963267948966) q[13];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[18];
sdg q[4];
ry(1.5707963267948966) q[18];
s q[2];
ry(1.5707963267948966) q[19];
u1(1.5707963267948966) q[20];
z q[19];
u3(0, 0, 1.5707963267948966) q[16];
h q[14];
t q[18];
y q[20];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[9];
t q[7];
rx(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[12];
h q[7];
rz(1.5707963267948966) q[14];
sdg q[0];
sdg q[7];
rx(1.5707963267948966) q[13];
y q[21];
t q[4];
tdg q[20];
u3(0, 0, 1.5707963267948966) q[17];
y q[7];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[14];
h q[20];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[21];
z q[17];
sdg q[15];
y q[0];
sdg q[7];
t q[15];
u3(0, 0, 1.5707963267948966) q[9];
t q[15];
u2(1.5707963267948966, 1.5707963267948966) q[2];
y q[2];
y q[1];
tdg q[6];
z q[8];
t q[10];
tdg q[7];
u3(0, 0, 1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[11];
x q[12];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[12];
z q[2];
y q[12];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[10];
sdg q[15];
ry(1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[14];
tdg q[10];
x q[18];
x q[2];
x q[10];
x q[9];
rz(1.5707963267948966) q[13];
y q[19];
ry(1.5707963267948966) q[6];
y q[18];
u3(0, 0, 1.5707963267948966) q[20];
s q[17];
s q[18];
u3(0, 0, 1.5707963267948966) q[16];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[15];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[20];
t q[5];
sdg q[17];
sdg q[12];
u2(1.5707963267948966, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[10];
z q[3];
y q[0];
sdg q[12];
y q[9];
sdg q[5];
y q[14];
z q[17];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[7];
t q[19];
h q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
h q[14];
z q[7];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[6];
h q[11];
h q[3];
sdg q[0];
x q[3];
tdg q[5];
tdg q[11];
rz(1.5707963267948966) q[21];
tdg q[5];
y q[17];
h q[11];
y q[19];
u2(1.5707963267948966, 1.5707963267948966) q[19];
t q[8];
h q[10];
u2(1.5707963267948966, 1.5707963267948966) q[21];
s q[13];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[9];
t q[15];
s q[10];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[20];
t q[20];
u2(1.5707963267948966, 1.5707963267948966) q[13];
rx(1.5707963267948966) q[6];
s q[17];
y q[7];
sdg q[1];
h q[14];
rz(1.5707963267948966) q[19];
y q[6];
x q[14];
s q[16];
u2(1.5707963267948966, 1.5707963267948966) q[9];
z q[11];
u3(0, 0, 1.5707963267948966) q[14];
u1(1.5707963267948966) q[8];
z q[9];
u2(1.5707963267948966, 1.5707963267948966) q[11];
h q[13];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[18];