OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
y q[9];
sdg q[7];
z q[10];
s q[1];
u1(1.5707963267948966) q[13];
tdg q[0];
rz(1.5707963267948966) q[6];
s q[8];
y q[12];
tdg q[16];
s q[2];
x q[10];
x q[16];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[13];
u1(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[10];
s q[5];
rx(1.5707963267948966) q[2];
y q[5];
tdg q[11];
u3(0, 0, 1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[3];
x q[5];
h q[11];
sdg q[2];
u1(1.5707963267948966) q[4];
ry(1.5707963267948966) q[15];
x q[15];
rz(1.5707963267948966) q[14];
sdg q[11];
z q[0];
rx(1.5707963267948966) q[10];
tdg q[0];
y q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
x q[2];
rz(1.5707963267948966) q[2];
sdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[12];
rx(1.5707963267948966) q[11];
h q[8];
h q[16];
y q[2];
rz(1.5707963267948966) q[5];
h q[14];
u3(0, 0, 1.5707963267948966) q[13];
t q[9];
h q[7];
tdg q[3];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[8];
y q[6];
rz(1.5707963267948966) q[17];
sdg q[3];
h q[11];
sdg q[14];
t q[10];
x q[3];
sdg q[11];
x q[14];
h q[0];
u2(1.5707963267948966, 1.5707963267948966) q[14];
rz(1.5707963267948966) q[16];
h q[2];
x q[6];
s q[3];
rx(1.5707963267948966) q[14];
z q[14];
rz(1.5707963267948966) q[5];
s q[9];
u1(1.5707963267948966) q[17];
t q[5];
s q[16];
tdg q[5];
z q[12];
x q[5];
y q[1];
sdg q[14];
x q[6];
s q[12];
z q[3];
tdg q[10];
t q[12];
rz(1.5707963267948966) q[14];
s q[15];
u3(0, 0, 1.5707963267948966) q[16];
u1(1.5707963267948966) q[13];
t q[11];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[15];
s q[4];
s q[15];
rx(1.5707963267948966) q[6];
tdg q[15];
z q[9];
rz(1.5707963267948966) q[10];
tdg q[4];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[17];
h q[13];
u1(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[13];
z q[3];
y q[7];
z q[3];
u3(0, 0, 1.5707963267948966) q[5];
tdg q[1];
y q[13];
y q[4];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[16];
y q[11];
u1(1.5707963267948966) q[17];
s q[7];
z q[4];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[14];
sdg q[17];
u2(1.5707963267948966, 1.5707963267948966) q[13];
z q[8];
y q[11];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
tdg q[0];
t q[0];
rz(1.5707963267948966) q[2];
s q[10];
s q[14];
sdg q[3];
tdg q[17];
u2(1.5707963267948966, 1.5707963267948966) q[6];
sdg q[17];
x q[8];
y q[3];
u3(0, 0, 1.5707963267948966) q[3];
z q[4];
sdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[16];
x q[17];
u3(0, 0, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[13];
z q[14];
ry(1.5707963267948966) q[17];
y q[10];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[12];
y q[4];
rx(1.5707963267948966) q[8];
sdg q[10];
x q[14];
rx(1.5707963267948966) q[0];
y q[8];
x q[12];
s q[2];
y q[16];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u1(1.5707963267948966) q[3];
tdg q[14];
sdg q[15];
h q[11];
ry(1.5707963267948966) q[10];
y q[11];
tdg q[17];
h q[4];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[14];
y q[5];
tdg q[3];
z q[7];
x q[9];
y q[1];
tdg q[13];
u3(0, 0, 1.5707963267948966) q[1];
y q[0];
sdg q[17];
rx(1.5707963267948966) q[4];
sdg q[9];
h q[3];
sdg q[4];
x q[2];
t q[5];
y q[15];
h q[3];
u3(0, 0, 1.5707963267948966) q[0];
tdg q[8];
tdg q[12];
u3(0, 0, 1.5707963267948966) q[8];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[14];
s q[15];
s q[9];
z q[16];
y q[12];
rx(1.5707963267948966) q[0];
t q[6];
rz(1.5707963267948966) q[3];
z q[4];
ry(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[17];
h q[11];
x q[13];
u2(1.5707963267948966, 1.5707963267948966) q[14];
t q[16];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[13];
tdg q[1];
y q[16];
u1(1.5707963267948966) q[11];
