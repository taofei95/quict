OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
u1(1.5707963267948966) q[17];
s q[17];
rz(1.5707963267948966) q[0];
y q[10];
z q[14];
y q[5];
tdg q[18];
t q[14];
y q[14];
h q[17];
t q[3];
h q[3];
u1(1.5707963267948966) q[15];
sdg q[8];
u3(0, 0, 1.5707963267948966) q[0];
y q[16];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[3];
h q[17];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[13];
y q[17];
tdg q[16];
h q[2];
y q[10];
u3(0, 0, 1.5707963267948966) q[15];
z q[6];
sdg q[13];
h q[11];
h q[18];
z q[5];
y q[6];
x q[0];
y q[0];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[7];
ry(1.5707963267948966) q[4];
y q[13];
s q[9];
u2(1.5707963267948966, 1.5707963267948966) q[17];
tdg q[0];
u1(1.5707963267948966) q[10];
ry(1.5707963267948966) q[9];
tdg q[12];
u3(0, 0, 1.5707963267948966) q[13];
s q[7];
y q[2];
y q[15];
h q[16];
sdg q[4];
h q[14];
sdg q[8];
h q[18];
u2(1.5707963267948966, 1.5707963267948966) q[17];
tdg q[18];
rz(1.5707963267948966) q[3];
z q[9];
u1(1.5707963267948966) q[16];
u1(1.5707963267948966) q[13];
t q[16];
u3(0, 0, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[9];
h q[3];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[18];
s q[15];
u1(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[12];
sdg q[11];
s q[4];
z q[12];
ry(1.5707963267948966) q[12];
tdg q[10];
sdg q[15];
x q[6];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u1(1.5707963267948966) q[7];
t q[9];
u2(1.5707963267948966, 1.5707963267948966) q[5];
t q[9];
tdg q[18];
rz(1.5707963267948966) q[12];
z q[18];
y q[9];
rx(1.5707963267948966) q[5];
t q[17];
tdg q[8];
h q[4];
u1(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[10];
sdg q[9];
h q[3];
u1(1.5707963267948966) q[18];
tdg q[7];
ry(1.5707963267948966) q[10];
h q[16];
sdg q[11];
z q[15];
x q[16];
rx(1.5707963267948966) q[8];
tdg q[6];
rx(1.5707963267948966) q[14];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[15];
tdg q[12];
x q[15];
s q[4];
u1(1.5707963267948966) q[5];
y q[1];
y q[9];
sdg q[15];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[14];
u1(1.5707963267948966) q[13];
z q[8];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[9];
ry(1.5707963267948966) q[11];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u1(1.5707963267948966) q[10];
s q[7];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[16];
rx(1.5707963267948966) q[7];
z q[14];
ry(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[8];
h q[17];
u1(1.5707963267948966) q[17];
u1(1.5707963267948966) q[4];
h q[9];
h q[14];
s q[8];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[14];
rz(1.5707963267948966) q[13];
tdg q[12];
h q[1];
x q[9];
s q[2];
tdg q[0];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[0];
h q[13];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[18];
ry(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[14];
sdg q[1];
s q[15];
u3(0, 0, 1.5707963267948966) q[13];
y q[3];
sdg q[13];
u1(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rx(1.5707963267948966) q[9];
t q[18];
u2(1.5707963267948966, 1.5707963267948966) q[18];
y q[4];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[2];
y q[3];
tdg q[4];
t q[7];
sdg q[18];
sdg q[18];
u1(1.5707963267948966) q[15];
sdg q[18];
tdg q[11];
s q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
tdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[12];
x q[1];
x q[16];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[11];
tdg q[12];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[7];
h q[0];
sdg q[6];
s q[18];
u3(0, 0, 1.5707963267948966) q[17];
t q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[13];
sdg q[2];
x q[4];
s q[0];
s q[15];
sdg q[16];
t q[14];
sdg q[18];
t q[13];
t q[1];
u3(0, 0, 1.5707963267948966) q[4];
y q[9];
x q[16];
s q[1];
y q[4];
x q[17];
u1(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[17];
x q[0];
u1(1.5707963267948966) q[2];
tdg q[8];
s q[17];
h q[9];
t q[11];
h q[17];
u1(1.5707963267948966) q[17];
