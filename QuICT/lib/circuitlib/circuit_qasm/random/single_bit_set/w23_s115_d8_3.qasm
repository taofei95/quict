OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
rx(1.5707963267948966) q[2];
h q[14];
h q[18];
z q[4];
u2(1.5707963267948966, 1.5707963267948966) q[15];
t q[1];
sdg q[3];
z q[4];
y q[4];
rx(1.5707963267948966) q[10];
x q[13];
s q[12];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[12];
rx(1.5707963267948966) q[20];
rz(1.5707963267948966) q[14];
x q[10];
rx(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[11];
y q[12];
t q[7];
t q[5];
u2(1.5707963267948966, 1.5707963267948966) q[8];
y q[16];
h q[12];
u1(1.5707963267948966) q[20];
rz(1.5707963267948966) q[21];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[18];
t q[4];
u3(0, 0, 1.5707963267948966) q[11];
sdg q[15];
h q[11];
h q[22];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[11];
sdg q[18];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
t q[1];
s q[19];
z q[3];
x q[22];
t q[10];
rz(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[16];
t q[5];
tdg q[21];
u2(1.5707963267948966, 1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[7];
rz(1.5707963267948966) q[10];
sdg q[12];
z q[13];
z q[11];
x q[14];
u3(0, 0, 1.5707963267948966) q[2];
h q[18];
ry(1.5707963267948966) q[0];
x q[21];
z q[21];
u3(0, 0, 1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[21];
t q[19];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[16];
z q[6];
ry(1.5707963267948966) q[0];
t q[5];
x q[5];
u1(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[19];
t q[4];
u2(1.5707963267948966, 1.5707963267948966) q[17];
rx(1.5707963267948966) q[0];
u1(1.5707963267948966) q[0];
tdg q[12];
rz(1.5707963267948966) q[17];
tdg q[14];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[5];
z q[10];
s q[22];
rz(1.5707963267948966) q[13];
sdg q[6];
ry(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[8];
y q[1];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[9];
y q[18];
h q[18];
y q[10];
sdg q[12];
u2(1.5707963267948966, 1.5707963267948966) q[14];
sdg q[21];
sdg q[0];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[11];
tdg q[7];
u1(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[21];
s q[3];
u1(1.5707963267948966) q[6];
t q[19];
t q[4];
rz(1.5707963267948966) q[5];
