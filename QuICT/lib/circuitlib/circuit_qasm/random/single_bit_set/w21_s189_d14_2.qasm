OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
s q[5];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[16];
s q[4];
rx(1.5707963267948966) q[19];
y q[6];
s q[4];
x q[15];
ry(1.5707963267948966) q[9];
u1(1.5707963267948966) q[18];
x q[13];
u2(1.5707963267948966, 1.5707963267948966) q[19];
tdg q[5];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[0];
x q[5];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[11];
h q[6];
tdg q[16];
u2(1.5707963267948966, 1.5707963267948966) q[1];
tdg q[10];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[9];
t q[18];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[20];
u1(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[9];
tdg q[5];
z q[16];
ry(1.5707963267948966) q[9];
t q[5];
u3(0, 0, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[20];
rz(1.5707963267948966) q[0];
x q[20];
tdg q[15];
rx(1.5707963267948966) q[20];
h q[19];
u3(0, 0, 1.5707963267948966) q[6];
h q[18];
t q[2];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[13];
h q[9];
rz(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[8];
y q[3];
ry(1.5707963267948966) q[8];
t q[17];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[20];
x q[11];
s q[19];
z q[16];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[4];
tdg q[2];
h q[16];
s q[20];
u1(1.5707963267948966) q[8];
ry(1.5707963267948966) q[19];
y q[2];
tdg q[15];
sdg q[12];
t q[2];
u3(0, 0, 1.5707963267948966) q[20];
sdg q[2];
z q[17];
rz(1.5707963267948966) q[2];
y q[12];
t q[14];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[15];
s q[2];
y q[17];
u3(0, 0, 1.5707963267948966) q[16];
tdg q[13];
u1(1.5707963267948966) q[8];
x q[16];
u3(0, 0, 1.5707963267948966) q[14];
tdg q[19];
h q[9];
x q[10];
y q[12];
rx(1.5707963267948966) q[8];
tdg q[14];
rx(1.5707963267948966) q[19];
tdg q[11];
z q[20];
tdg q[12];
rz(1.5707963267948966) q[15];
y q[19];
y q[9];
t q[10];
u1(1.5707963267948966) q[12];
tdg q[10];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[17];
tdg q[12];
u1(1.5707963267948966) q[0];
s q[11];
s q[9];
tdg q[8];
rx(1.5707963267948966) q[17];
z q[18];
z q[4];
rz(1.5707963267948966) q[0];
h q[17];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[20];
u2(1.5707963267948966, 1.5707963267948966) q[10];
t q[2];
u2(1.5707963267948966, 1.5707963267948966) q[13];
z q[7];
tdg q[15];
x q[12];
sdg q[18];
u3(0, 0, 1.5707963267948966) q[13];
t q[14];
x q[14];
t q[18];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u1(1.5707963267948966) q[0];
x q[9];
u2(1.5707963267948966, 1.5707963267948966) q[10];
sdg q[16];
h q[3];
z q[0];
rz(1.5707963267948966) q[5];
s q[7];
u3(0, 0, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[13];
y q[19];
z q[10];
tdg q[4];
h q[8];
y q[8];
ry(1.5707963267948966) q[9];
sdg q[13];
sdg q[17];
x q[1];
sdg q[5];
z q[14];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[9];
t q[8];
z q[17];
x q[13];
h q[10];
h q[9];
rx(1.5707963267948966) q[0];
y q[18];
h q[11];
z q[14];
u3(0, 0, 1.5707963267948966) q[14];
u1(1.5707963267948966) q[8];
tdg q[5];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[10];
y q[12];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[9];
h q[18];
y q[19];
rz(1.5707963267948966) q[20];
z q[14];
h q[14];
s q[6];
u1(1.5707963267948966) q[13];
x q[5];
sdg q[12];
z q[16];
