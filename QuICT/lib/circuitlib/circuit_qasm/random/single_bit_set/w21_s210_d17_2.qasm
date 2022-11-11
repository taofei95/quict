OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
u3(0, 0, 1.5707963267948966) q[11];
u1(1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[16];
y q[10];
y q[18];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[10];
rx(1.5707963267948966) q[16];
sdg q[5];
tdg q[15];
sdg q[17];
s q[9];
sdg q[14];
u1(1.5707963267948966) q[18];
h q[5];
rx(1.5707963267948966) q[20];
rx(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[18];
rx(1.5707963267948966) q[17];
s q[15];
y q[7];
h q[2];
u1(1.5707963267948966) q[6];
x q[12];
sdg q[16];
sdg q[8];
sdg q[10];
t q[5];
rz(1.5707963267948966) q[16];
tdg q[14];
x q[19];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[16];
z q[3];
h q[17];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
u1(1.5707963267948966) q[1];
sdg q[8];
u1(1.5707963267948966) q[18];
t q[14];
u2(1.5707963267948966, 1.5707963267948966) q[16];
s q[2];
rz(1.5707963267948966) q[15];
z q[6];
ry(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[16];
s q[13];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[16];
s q[9];
rz(1.5707963267948966) q[13];
y q[3];
u1(1.5707963267948966) q[17];
z q[13];
z q[7];
rx(1.5707963267948966) q[13];
y q[15];
rz(1.5707963267948966) q[3];
sdg q[8];
u3(0, 0, 1.5707963267948966) q[19];
s q[11];
tdg q[4];
tdg q[2];
sdg q[5];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[13];
x q[0];
u1(1.5707963267948966) q[20];
x q[16];
sdg q[17];
sdg q[15];
x q[18];
u1(1.5707963267948966) q[11];
x q[13];
sdg q[19];
rz(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[12];
rx(1.5707963267948966) q[8];
y q[0];
rx(1.5707963267948966) q[20];
rz(1.5707963267948966) q[15];
h q[7];
u3(0, 0, 1.5707963267948966) q[20];
y q[15];
u1(1.5707963267948966) q[17];
t q[14];
y q[7];
z q[11];
u2(1.5707963267948966, 1.5707963267948966) q[20];
tdg q[12];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[19];
rx(1.5707963267948966) q[5];
t q[12];
h q[5];
rx(1.5707963267948966) q[7];
s q[8];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[10];
u1(1.5707963267948966) q[9];
t q[13];
y q[12];
sdg q[8];
u3(0, 0, 1.5707963267948966) q[14];
z q[0];
x q[4];
sdg q[17];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[6];
tdg q[18];
y q[2];
z q[4];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
z q[19];
y q[20];
u2(1.5707963267948966, 1.5707963267948966) q[14];
s q[16];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[8];
z q[20];
tdg q[16];
u3(0, 0, 1.5707963267948966) q[6];
y q[16];
sdg q[16];
u1(1.5707963267948966) q[8];
ry(1.5707963267948966) q[14];
h q[20];
x q[12];
ry(1.5707963267948966) q[1];
h q[0];
y q[12];
h q[7];
u3(0, 0, 1.5707963267948966) q[12];
y q[0];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[18];
sdg q[20];
x q[4];
u1(1.5707963267948966) q[13];
x q[14];
tdg q[19];
s q[14];
s q[19];
y q[1];
sdg q[18];
u3(0, 0, 1.5707963267948966) q[3];
t q[19];
rz(1.5707963267948966) q[0];
z q[20];
tdg q[20];
u2(1.5707963267948966, 1.5707963267948966) q[15];
sdg q[0];
s q[1];
u1(1.5707963267948966) q[18];
tdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
s q[7];
sdg q[14];
y q[14];
rz(1.5707963267948966) q[20];
sdg q[19];
sdg q[8];
u3(0, 0, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[8];
sdg q[4];
rx(1.5707963267948966) q[8];
x q[17];
u3(0, 0, 1.5707963267948966) q[11];
u1(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[0];
sdg q[0];
ry(1.5707963267948966) q[12];
u1(1.5707963267948966) q[12];
ry(1.5707963267948966) q[3];
x q[2];
u1(1.5707963267948966) q[16];
h q[11];
sdg q[9];
s q[13];
u1(1.5707963267948966) q[15];
tdg q[3];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[18];
ry(1.5707963267948966) q[8];
t q[11];
z q[8];
u1(1.5707963267948966) q[6];
y q[10];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[2];
x q[2];
u2(1.5707963267948966, 1.5707963267948966) q[19];
x q[12];
y q[8];
