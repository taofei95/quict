OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
tdg q[0];
s q[10];
rx(1.5707963267948966) q[12];
t q[10];
rx(1.5707963267948966) q[13];
u1(1.5707963267948966) q[5];
sdg q[14];
tdg q[8];
ry(1.5707963267948966) q[7];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[20];
z q[12];
u3(0, 0, 1.5707963267948966) q[4];
h q[0];
h q[4];
u1(1.5707963267948966) q[7];
tdg q[10];
rz(1.5707963267948966) q[14];
h q[13];
y q[5];
h q[9];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[19];
sdg q[8];
y q[10];
rz(1.5707963267948966) q[13];
s q[18];
tdg q[9];
u3(0, 0, 1.5707963267948966) q[17];
h q[15];
y q[20];
sdg q[7];
x q[2];
s q[8];
tdg q[3];
tdg q[3];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[14];
tdg q[19];
h q[9];
t q[9];
x q[8];
u1(1.5707963267948966) q[19];
t q[1];
x q[18];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
z q[13];
u3(0, 0, 1.5707963267948966) q[9];
u1(1.5707963267948966) q[11];
y q[18];
rz(1.5707963267948966) q[15];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[15];
y q[12];
s q[4];
y q[16];
x q[4];
tdg q[9];
h q[17];
t q[17];
s q[12];
t q[5];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[19];
sdg q[12];
tdg q[12];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[19];
sdg q[17];
s q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
x q[11];
u2(1.5707963267948966, 1.5707963267948966) q[4];
sdg q[6];
z q[17];
h q[11];
rz(1.5707963267948966) q[17];
t q[9];
sdg q[1];
rz(1.5707963267948966) q[20];
t q[13];
tdg q[5];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[8];
t q[18];
s q[10];
h q[19];
s q[1];
t q[3];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[2];
h q[1];
y q[14];
h q[17];
tdg q[6];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[1];
sdg q[1];
x q[17];
z q[15];
z q[20];
h q[1];
ry(1.5707963267948966) q[13];
tdg q[2];
x q[13];
z q[11];
u1(1.5707963267948966) q[12];
sdg q[11];
u2(1.5707963267948966, 1.5707963267948966) q[0];
z q[14];
t q[19];
x q[17];
x q[10];
y q[0];
t q[8];
rx(1.5707963267948966) q[18];
z q[14];
rx(1.5707963267948966) q[11];
sdg q[16];
u3(0, 0, 1.5707963267948966) q[17];
z q[15];
z q[18];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[13];
y q[0];
ry(1.5707963267948966) q[16];
sdg q[14];
h q[4];
tdg q[13];
u3(0, 0, 1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[1];
sdg q[8];
z q[18];
sdg q[17];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[19];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[15];
t q[8];
rx(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[11];
s q[14];
x q[3];
s q[20];
rx(1.5707963267948966) q[17];
y q[10];
sdg q[0];
rz(1.5707963267948966) q[11];
sdg q[20];
ry(1.5707963267948966) q[15];
z q[7];
u1(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[20];
t q[17];
u3(0, 0, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[2];
x q[10];
u3(0, 0, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[20];
x q[20];
tdg q[2];
h q[16];
z q[11];
